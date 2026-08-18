#!/usr/bin/env python3
"""Cinematic 'commander POV' trailer over a formation of N Unitree G1 humanoids.

Reuses the project's existing assets end to end: the official G1 MJCF and meshes via
scripts/build_army_scene.py, the retargeted Kalari motion library in
data/motions_retargeted/ for the poses, and the MuJoCo renderer the review pipeline
already uses. Nothing about the robot is redesigned.

Pipeline
    1. pick N motion clips whose poses are as visually unlike each other as possible
    2. lay the robots out in staggered ranks, spacing widening toward the rear so the
       back ranks stay visible from an elevated eye
    3. drive one continuous camera along a time-parameterised Catmull-Rom spline
       through establish -> descend -> travel/showcase -> pull back -> final wide
    4. render RGB and depth each frame
    5. composite depth-of-field, bloom, grade, vignette and a 2.39:1 letterbox

What is real and what is faked: MuJoCo gives true shadow maps, multisampling and
horizon haze. It has no volumetrics and no lens model, so depth of field is computed
from the real depth buffer in post, and bloom/grade/vignette are 2D passes. There is
no ray tracing here; this is a rasteriser dressed for camera.

Usage
    python3 scripts/render_army_trailer.py                 # full 60s
    python3 scripts/render_army_trailer.py --duration 8 --fps 12 --width 960 --preview
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mujoco

from scripts.render_g1_motion import load_npz_motion
from src.sim.conventions import quat_xyzw_to_wxyz
from src.sim.mujoco_runtime import select_gl_backend

MOTION_DIR = "data/motions_retargeted"
SCENE = "assets/unitree_g1/army_576.xml"
N_ROBOTS = 576
POSE_PLAYBACK = 0.16       # slow oscillation around a held stance, not clip playback


# ------------------------------------------------------------------ pose picking
def combat_window(motion: dict) -> tuple[int, int]:
    """Longest stretch where the robot is upright and on its feet.

    The Kalari library is full of ground work: sweeps, floor salutations, deep
    prone stretches. Played back untouched, a third of the formation ends up lying
    on the parade ground, which is not a combat-ready army. Restricting playback to
    an upright window keeps every robot standing and fighting.
    """
    from src.sim.conventions import quat_wxyz_to_matrix
    q = motion["root_quat_xyzw"]
    z = motion["root_pos"][:, 2]
    up = np.array([quat_wxyz_to_matrix(quat_xyzw_to_wxyz(x))[2, 2] for x in q])
    # 0.80 not 0.72: at the looser threshold a diving side-kick still qualifies and
    # reads as a robot falling over when it is one of thirty in a parade formation.
    ok = (up > 0.80) & (z > max(0.55, 0.62 * float(np.max(z))))
    best, cur = (0, len(ok)), None
    runs = []
    for i, v in enumerate(ok):
        if v and cur is None:
            cur = i
        elif not v and cur is not None:
            runs.append((cur, i)); cur = None
    if cur is not None:
        runs.append((cur, len(ok)))
    runs = [r for r in runs if r[1] - r[0] >= 12]
    if runs:
        best = max(runs, key=lambda r: r[1] - r[0])
    return best


def pose_descriptor(motion: dict) -> np.ndarray:
    """Signature pose: the most distinctive frame *within* the upright window.

    Taking the extreme frame over the whole clip would pick a floor pose almost
    every time, so the diversity search would be choosing between ways of lying
    down rather than between fighting stances.
    """
    k0, k1 = combat_window(motion)
    j = motion["joint_pos"][k0:k1]
    if len(j) == 0:
        j = motion["joint_pos"]
    d = np.linalg.norm(j - j.mean(axis=0, keepdims=True), axis=1)
    return j[int(np.argmax(d))]


def hold_window(motion: dict, half: int = 11) -> tuple[int, int]:
    """A narrow band around the clip's signature pose.

    The formation reads as an army holding combat stances, not as thirty clips
    playing at once. Replaying whole clips gave every robot a full Kalari sequence,
    so the ranks churned and individual silhouettes never resolved. Oscillating
    across roughly two thirds of a second around one signature frame keeps each
    robot's identity legible while still breathing.
    """
    k0, k1 = combat_window(motion)
    j = motion["joint_pos"][k0:k1]
    if len(j) < 3:
        k0, k1 = 0, len(motion["joint_pos"])
        j = motion["joint_pos"]
    sig = k0 + int(np.argmax(np.linalg.norm(j - j.mean(axis=0, keepdims=True), axis=1)))
    lo = max(k0, sig - half)
    hi = min(k1, sig + half)
    if hi - lo < 3:
        lo, hi = k0, min(k1, k0 + 3)
    return lo, hi


def pick_diverse(n: int, exclude: set[str]) -> list[str]:
    """Greedy farthest-point selection over signature poses."""
    ids, feats = [], []
    for f in sorted(os.listdir(MOTION_DIR)):
        if not f.endswith(".npz"):
            continue
        mid = f[:-4]
        if mid in exclude or mid.startswith("test_"):
            continue
        ids.append(mid)
        feats.append(pose_descriptor(load_npz_motion(os.path.join(MOTION_DIR, f))))
    X = np.array(feats)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    chosen = [int(np.argmax(np.linalg.norm(X - X.mean(0), axis=1)))]
    while len(chosen) < min(n, len(ids)):
        d = np.min(np.linalg.norm(X[:, None, :] - X[chosen][None, :, :], axis=2), axis=1)
        d[chosen] = -1
        chosen.append(int(np.argmax(d)))
    out = [ids[i] for i in chosen]
    while len(out) < n:                      # library smaller than the formation
        out.append(ids[len(out) % len(ids)])
    return out[:n]


# ------------------------------------------------------------------ formation
# Three training boxes, mirroring build_army_scene.ZONES. Ten humanoids each.
ZONE_CX = {"beginner": -12.0, "defense": 0.0, "attack": 12.0}
ZONE_ORDER = ["beginner", "defense", "attack"]
ZONE_TITLE = {"beginner": "BEGINNER LEVEL", "defense": "DEFENSE LEVEL",
              "attack": "ATTACK LEVEL"}
# Label pools, cycled per zone. Kept short: the HUD is an analysis readout, not a
# caption track.
ZONE_LABELS = {
    "beginner": ["BASIC GUARD", "FORWARD STANCE", "BASIC STRIKE",
                 "BALANCE TRAINING", "READY POSITION"],
    "defense":  ["DEFENSIVE GUARD", "BLOCK", "EVASION",
                 "COUNTER POSITION", "LOW DEFENSE"],
    "attack":   ["FORWARD STRIKE", "POWER STRIKE", "ATTACK STANCE",
                 "CHARGING POSITION", "ADVANCED STRIKE"],
}


def formation(n: int):
    """Ten robots per zone in staggered 5x2 ranks, all facing the commander.

    Returns (positions, yaws, zone-key per robot). Rows are offset half a pitch so
    the rear rank is never hidden behind the front one from an elevated eye, which
    is the same trick the single-block version used, applied per box.
    """
    # Nine separate 8x8 troupes in a 3x3 grid: 576 humanoids, one arena.
    # Columns carry the training level (left=beginner, centre=defense,
    # right=attack) so clip families stay coherent without any floor markings.
    pos, yaw, zone = [], [], []
    rng = np.random.default_rng(7)
    SP = 1.65                                  # rank/file pitch inside a troupe
    COLS = {-17.0: "beginner", 0.0: "defense", 17.0: "attack"}
    ROWS_Y = [2.0, 19.0, 36.0]                 # troupe front edges, 3 deep
    for by in ROWS_Y:
        for bx, zk in COLS.items():
            for i in range(64):
                r, c = divmod(i, 8)
                x = bx + (c - 3.5) * SP
                y = by + r * SP
                pos.append([x, y, 0.0])
                yaw.append(-90.0 + float(rng.uniform(-2.0, 2.0)))  # -90 faces -y
                zone.append(zk)
    return np.array(pos, float), np.array(yaw, float), zone


def classify(mid: str, name_hint: str = "") -> str:
    """Route a clip to a training zone by what the move actually is."""
    n = (mid + " " + name_hint).lower()
    if "block" in n or ("guard" in n and "kick" not in n):
        return "defense"
    if "kick" in n or "strike" in n or "jump" in n or "leap" in n:
        return "attack"
    return "beginner"


def pick_zone_clips(per_zone: int, exclude: set[str]) -> dict[str, list[str]]:
    """Diverse clips within each zone, so silhouettes differ inside a box too."""
    buckets: dict[str, list[tuple[str, np.ndarray]]] = {z: [] for z in ZONE_ORDER}
    for f in sorted(os.listdir(MOTION_DIR)):
        if not f.endswith(".npz"):
            continue
        mid = f[:-4]
        if mid in exclude or mid.startswith("test_"):
            continue
        mo = load_npz_motion(os.path.join(MOTION_DIR, f))
        buckets[classify(mid)].append((mid, pose_descriptor(mo)))
    out: dict[str, list[str]] = {}
    for zk in ZONE_ORDER:
        items = buckets[zk]
        if not items:
            items = [(m, d) for b in buckets.values() for m, d in b]
        X = np.array([d for _, d in items])
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        chosen = [int(np.argmax(np.linalg.norm(X - X.mean(0), axis=1)))]
        while len(chosen) < min(per_zone, len(items)):
            dd = np.min(np.linalg.norm(X[:, None] - X[chosen][None], axis=2), axis=1)
            dd[chosen] = -1
            chosen.append(int(np.argmax(dd)))
        ids = [items[i][0] for i in chosen]
        while len(ids) < per_zone:                    # small bucket: cycle it
            ids.append(ids[len(ids) % max(1, len(chosen))])
        out[zk] = ids[:per_zone]
    return out


# ------------------------------------------------------------------ camera spline
def catmull(P: np.ndarray, t: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Catmull-Rom through control points P sampled at times t, evaluated at q."""
    out = np.zeros((len(q), P.shape[1]))
    for k, tq in enumerate(q):
        i = int(np.clip(np.searchsorted(t, tq) - 1, 0, len(t) - 2))
        u = (tq - t[i]) / max(t[i + 1] - t[i], 1e-9)
        p0 = P[max(i - 1, 0)]; p1 = P[i]; p2 = P[i + 1]; p3 = P[min(i + 2, len(P) - 1)]
        u2, u3 = u * u, u * u * u
        out[k] = 0.5 * ((2 * p1) + (-p0 + p2) * u
                        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2
                        + (-p0 + 3 * p1 - 3 * p2 + p3) * u3)
    return out


def orbit_points(centre: np.ndarray, radius: float, h: float,
                 a0: float, a1: float, k: int) -> list[np.ndarray]:
    return [centre + np.array([radius * np.cos(a), radius * np.sin(a), h])
            for a in np.linspace(a0, a1, k)]


def camera_track(pos: np.ndarray, zone: list[str], duration: float, fps: int,
                 heroes_per_zone: int = 3):
    """One continuous commander move across the three training boxes.

    Returns (eye, target, fov, annotations). The annotation schedule is emitted from
    the same control points that steer the camera, so a label can never appear for a
    robot the camera is not actually looking at: the two cannot drift out of sync
    because they are the same data.
    """
    # Two hero showcases, slow heavy camera, bookended wides (final brief):
    # 0-5 establishing / 5-10 approach / 10-14 isolate hero 1 / 14-18 action 1
    # 18-21 lateral transition / 21-25 action 2 / 25-30 pull-back echo.
    def nearest(px, py, zk):
        best, bi = 1e18, 0
        for i, z in enumerate(zone):
            if z != zk:
                continue
            d = (pos[i][0] - px) ** 2 + (pos[i][1] - py) ** 2
            if d < best:
                best, bi = d, i
        return bi

    h1 = nearest(0.8, 2.0, "defense")
    h2 = nearest(16.5, 2.0, "attack")
    p1, p2 = pos[h1], pos[h2]

    K: list[tuple[float, list, list, float]] = []
    add = lambda t, e, tg, f: K.append((t, list(e), list(tg), f))
    anno: list[dict] = []

    add(0.0, [-13.0, -26.0, 13.0], [0.0, 24.0, 1.5], 37.0)
    add(5.0, [-11.0, -23.0, 12.0], [0.0, 22.0, 1.5], 37.0)
    add(10.0, [-5.0, -14.0, 6.0], [0.0, 8.0, 1.4], 39.0)
    add(14.0, [p1[0] - 2.4, p1[1] - 5.2, 2.6], [p1[0], p1[1], 1.1], 38.0)
    add(18.0, [p1[0] - 1.6, p1[1] - 4.6, 2.2], [p1[0], p1[1], 1.05], 36.0)
    anno.append({"robot": int(h1), "zone": zone[h1], "label": "", "title": "",
                 "t0": 13.5, "t1": 17.8})
    add(21.0, [p2[0] - 4.5, p2[1] - 7.5, 3.4], [p2[0], p2[1], 1.2], 40.0)
    add(25.0, [p2[0] - 1.8, p2[1] - 4.8, 2.1], [p2[0], p2[1], 1.05], 36.0)
    anno.append({"robot": int(h2), "zone": zone[h2], "label": "", "title": "",
                 "t0": 20.8, "t1": 24.8})
    add(28.0, [-10.0, -22.0, 10.0], [0.0, 20.0, 2.0], 38.0)
    add(30.0, [-13.0, -27.0, 13.5], [0.0, 24.0, 1.8], 37.0)

    K.sort(key=lambda r: r[0])
    scale = duration / 30.0
    tk = np.array([k[0] for k in K]) * scale
    eye = np.array([k[1] for k in K]); tgt = np.array([k[2] for k in K])
    fov = np.array([[k[3]] for k in K])
    for a in anno:
        a["t0"] *= scale; a["t1"] *= scale

    n = int(round(duration * fps))
    q = np.linspace(0.0, duration, n)
    E, T, F = catmull(eye, tk, q), catmull(tgt, tk, q), catmull(fov, tk, q)[:, 0]

    sig = max(1.0, 0.28 * fps)
    r = int(sig * 3)
    g = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2); g /= g.sum()
    sm = lambda A: np.stack([np.convolve(np.pad(A[:, i], (r, r), mode="edge"), g,
                                         mode="same")[r:-r] for i in range(A.shape[1])], 1)
    return sm(E), sm(T), sm(F[:, None])[:, 0], anno


def qmul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


def clip_heading_deg(motion: dict) -> float:
    """Mean facing of a clip, so it can be re-aimed onto its formation slot."""
    from src.sim.conventions import quat_wxyz_to_matrix
    v = np.zeros(2)
    q = motion["root_quat_xyzw"]
    for x in q[:: max(1, len(q) // 40)]:
        v += quat_wxyz_to_matrix(quat_xyzw_to_wxyz(x))[:2, 0]
    return float(np.degrees(np.arctan2(v[1], v[0])))


def look_at(eye: np.ndarray, target: np.ndarray):
    """MuJoCo free-camera parameters that place the eye exactly at `eye`."""
    d = target - eye
    dist = float(np.linalg.norm(d))
    f = d / max(dist, 1e-9)
    az = float(np.degrees(np.arctan2(f[1], f[0])))
    el = float(np.degrees(np.arcsin(np.clip(f[2], -1, 1))))
    return dist, az, el


# ------------------------------------------------------------------ HUD
def project(pt: np.ndarray, eye: np.ndarray, target: np.ndarray,
            fovy_deg: float, w: int, h: int):
    """World point -> pixel in the FULL render, plus depth along the view axis.

    Built from the same eye/target/fovy the renderer was handed, so the marker lands
    exactly on the robot rather than approximately near it. Returns None when the
    point is behind the camera, which otherwise projects to a mirrored ghost.
    """
    fwd = target - eye
    fwd = fwd / max(np.linalg.norm(fwd), 1e-9)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    rel = pt - eye
    z = float(np.dot(rel, fwd))
    if z <= 0.05:
        return None
    f = 1.0 / np.tan(np.radians(fovy_deg) / 2.0)
    ndc_x = (np.dot(rel, right) / z) * f / (w / h)
    ndc_y = (np.dot(rel, up) / z) * f
    return (int(round((ndc_x * 0.5 + 0.5) * w)),
            int(round((0.5 - ndc_y * 0.5) * h)), z)


def draw_hud(img: np.ndarray, px: int, py: int, label: str, title: str,
             alpha: float, crop_off: int) -> np.ndarray:
    """Thin leader line from the humanoid up to a small floating readout.

    Drawn on the full frame before the letterbox crop, with the crop offset applied,
    so the anchor stays welded to the robot. Everything is alpha-blended through one
    overlay buffer so the whole annotation fades as a unit instead of the line and
    the text popping at different moments.
    """
    if alpha <= 0.01:
        return img
    h, w = img.shape[:2]
    py -= crop_off
    if not (-200 < px < w + 200 and -200 < py < h + 200):
        return img
    ov = img.copy()
    # Dark ink on a white arena: the previous near-white HUD vanished against the floor.
    ink = (26, 32, 44)
    accent = (196, 132, 24)
    k = w / 1920.0                    # scale every dimension with the frame
    lw_line = max(1, int(round(2 * k)))
    rise = int(150 * k)
    elbow_y = max(int(34 * k), py - rise)
    side = 1 if px < w * 0.62 else -1
    run = int(180 * k) * side
    cv2.circle(ov, (px, py), max(2, int(6 * k)), accent, -1, cv2.LINE_AA)
    cv2.circle(ov, (px, py), max(4, int(13 * k)), accent, lw_line, cv2.LINE_AA)
    cv2.line(ov, (px, py - int(14 * k)), (px, elbow_y), ink, lw_line, cv2.LINE_AA)
    cv2.line(ov, (px, elbow_y), (px + run, elbow_y), ink, lw_line, cv2.LINE_AA)
    tx = px + run + int(14 * k) * side
    fs_l, fs_t = 0.95 * k, 0.60 * k
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fs_l, 1)
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, fs_t, 1)
    bw = max(lw, tw)
    x0 = tx if side > 0 else tx - bw
    cv2.line(ov, (x0 - int(12 * k), elbow_y - int(30 * k)),
             (x0 - int(12 * k), elbow_y + int(38 * k)), accent,
             max(2, int(3 * k)), cv2.LINE_AA)
    cv2.putText(ov, title, (x0, elbow_y - int(10 * k)), cv2.FONT_HERSHEY_SIMPLEX,
                fs_t, accent, max(1, int(2 * k)), cv2.LINE_AA)
    cv2.putText(ov, label, (x0, elbow_y + int(30 * k)), cv2.FONT_HERSHEY_DUPLEX,
                fs_l, ink, max(1, int(2 * k)), cv2.LINE_AA)
    return cv2.addWeighted(ov, alpha, img, 1.0 - alpha, 0.0)


def anno_alpha(t: float, a: dict, fade: float = 0.45) -> float:
    """Ease in and out so labels never pop."""
    if t < a["t0"] - fade or t > a["t1"] + fade:
        return 0.0
    if t < a["t0"]:
        u = (t - (a["t0"] - fade)) / fade
    elif t > a["t1"]:
        u = 1.0 - (t - a["t1"]) / fade
    else:
        u = 1.0
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3 - 2 * u)


# ------------------------------------------------------------------ post
def depth_of_field(rgb: np.ndarray, depth: np.ndarray, focus: float,
                   strength: float = 1.0) -> np.ndarray:
    """Blend sharp / medium / heavy blurs by circle-of-confusion from real depth."""
    coc = np.abs(depth - focus) / max(focus, 1e-3)
    coc = np.clip(coc * 1.15 * strength, 0.0, 1.0)[..., None]
    med = cv2.GaussianBlur(rgb, (0, 0), 2.2)
    far = cv2.GaussianBlur(rgb, (0, 0), 6.5)
    a = np.clip(coc * 2.0, 0, 1)
    b = np.clip((coc - 0.5) * 2.0, 0, 1)
    return (rgb * (1 - a) + med * a) * (1 - b) + far * b


def bloom(img: np.ndarray, thresh: float = 0.86, gain: float = 0.16) -> np.ndarray:
    lum = img.mean(axis=2, keepdims=True)
    hi = np.clip((img - thresh * 255.0) * (lum > thresh * 255.0), 0, None)
    return img + gain * cv2.GaussianBlur(hi, (0, 0), 9.0)


def grade(img: np.ndarray) -> np.ndarray:
    """Cool shadows, warm highlights, lifted contrast, vignette."""
    x = img / 255.0
    # Gentle: the G1 meshes are near-white, so any real contrast push clips them
    # to flat silhouettes and the panel detail that sells "metallic" is gone.
    x = np.clip((x - 0.5) * 1.04 + 0.5, 0, 1)
    x = np.clip(x * 0.99 + 0.015, 0, 1)          # keep the white floor white
    # near-neutral: a teal/orange push would tint the whole floor and the brief is
    # explicit that the white has to stay white
    shadow = np.array([0.98, 0.99, 1.03]); high = np.array([1.02, 1.005, 0.985])
    w = x.mean(axis=2, keepdims=True)
    x = np.clip(x * (shadow * (1 - w) + high * w), 0, 1)
    h, wd = x.shape[:2]
    yy, xx = np.mgrid[0:h, 0:wd]
    r = np.sqrt(((xx - wd / 2) / (wd / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    x *= np.clip(1.0 - 0.16 * np.clip(r - 0.70, 0, None) ** 1.7, 0, 1)[..., None]
    return x * 255.0


def letterbox(img: np.ndarray, aspect: float = 2.39) -> np.ndarray:
    """Crop to a scope aspect, keeping both dimensions even.

    libx264 rejects odd dimensions outright, and 1920/2.39 lands on 803. That kills
    the encoder on the first frame with a broken pipe, which is a confusing way to
    discover a rounding bug an hour into a render.
    """
    h, w = img.shape[:2]
    keep = int(round(w / aspect)) & ~1          # force even
    if keep >= h:
        return img[: h & ~1, : w & ~1]
    off = (h - keep) // 2
    return img[off:off + keep, : w & ~1]


# ------------------------------------------------------------------ main render
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=SCENE)
    ap.add_argument("--out", default="results_sequences/army_trailer.mp4")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--ss", type=float, default=1.0,
                    help="supersample factor: render at ss*size then box-filter down. "
                         "MSAA fixes geometry edges but not shader/texture aliasing; "
                         "ss=1.5-2 is what removes the last of the crawl on the "
                         "banner artwork and the thin zone lines.")
    ap.add_argument("--crf", type=int, default=15,
                    help="x264 quality, lower is better. 15 is visually transparent; "
                         "the imageio default was leaving mosquito noise on the flat "
                         "white floor, which reads as 'pixelated'.")
    ap.add_argument("--preview", action="store_true", help="skip DOF/bloom for a fast look")
    ap.add_argument("--no-post", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.scene):
        raise SystemExit(f"{args.scene} not found. Run scripts/build_army_scene.py first.")

    select_gl_backend()
    m = mujoco.MjModel.from_xml_path(args.scene)
    d = mujoco.MjData(m)

    pos, yaw, zone = formation(N_ROBOTS)
    zone_clips = pick_zone_clips(N_ROBOTS // len(ZONE_ORDER),
                                 exclude={"sc_low_squat_drill"})
    counters = {z: 0 for z in ZONE_ORDER}
    clips = []
    for zk in zone:
        clips.append(zone_clips[zk][counters[zk] % len(zone_clips[zk])])
        counters[zk] += 1
    motions = [load_npz_motion(os.path.join(MOTION_DIR, f"{c}.npz")) for c in clips]
    print("zones: " + " | ".join(f"{z} {len(set(zone_clips[z]))} clips" for z in ZONE_ORDER))

    # qpos addressing + joint mapping, resolved once
    base_adr, jmap = [], []
    for i in range(N_ROBOTS):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"r{i:02d}_floating_base_joint")
        base_adr.append(m.jnt_qposadr[jid])
        names = list(motions[i]["joint_names"] or [])
        cols = {}
        for k, jn in enumerate(names):
            j2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"r{i:02d}_{jn}")
            if j2 >= 0:
                cols[m.jnt_qposadr[j2]] = k
        jmap.append(cols)
    print(f"formation {N_ROBOTS} robots | joints mapped per robot: "
          f"{[len(j) for j in jmap][:3]}... | distinct clips: {len(set(clips))}")

    # Per-robot placement transforms, resolved once: the yaw that turns each clip's
    # own mean facing onto its slot facing, the matching 2-D rotation for its travel,
    # and the clip centre that travel is measured from.
    yaw_q, rot2, clip_centre, windows = [], [], [], []
    for i in range(N_ROBOTS):
        dy = np.radians(yaw[i] - clip_heading_deg(motions[i]))
        yaw_q.append(np.array([np.cos(dy / 2), 0.0, 0.0, np.sin(dy / 2)]))
        c, s = np.cos(dy), np.sin(dy)
        rot2.append(np.array([[c, -s], [s, c]]))
        w = hold_window(motions[i])
        windows.append(w)
        clip_centre.append(motions[i]["root_pos"][w[0]:w[1], :2].mean(axis=0))
    print(f"stance hold windows: median {int(np.median([w[1]-w[0] for w in windows]))} "
          f"frames, shortest {min(w[1]-w[0] for w in windows)}")

    E, T, F, annos = camera_track(pos, zone, args.duration, args.fps)
    nframes = len(E)
    phase = np.random.default_rng(3).uniform(0, 1, N_ROBOTS)

    RW = int(round(args.width * args.ss)) & ~1
    RH = int(round(args.height * args.ss)) & ~1
    if RW > m.vis.global_.offwidth or RH > m.vis.global_.offheight:
        raise SystemExit(f"offscreen buffer is {m.vis.global_.offwidth}x"
                         f"{m.vis.global_.offheight}, need {RW}x{RH}; raise it in "
                         f"scripts/build_army_scene.py cinematic_visual()")
    print(f"rendering at {RW}x{RH}, delivering {args.width}x~{int(args.width/2.39)} "
          f"(ss={args.ss}, crf={args.crf})")
    r = mujoco.Renderer(m, RH, RW)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    # Stream to the encoder instead of collecting frames: 1800 frames of 1920x804
    # RGB is about 8 GB resident, which this machine would swap or die on.
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    import imageio.v2 as imageio
    writer = imageio.get_writer(
        args.out, fps=args.fps, codec="libx264", macro_block_size=None,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", str(args.crf),
                       "-preset", "slow", "-tune", "film",
                       "-x264-params", "aq-mode=3:psy-rd=1.0",
                       "-movflags", "+faststart"])
    out_shape = None
    t0 = time.time()
    for f in range(nframes):
        tsec = f / args.fps
        for i in range(N_ROBOTS):
            mo = motions[i]
            k0, k1 = windows[i]
            span = max(1, k1 - k0 - 1)
            src_fps = float(mo["fps"] or 30.0)
            # ping-pong inside the upright window so a looping clip never snaps back
            # to frame 0 on camera, and never drops to the floor mid-formation
            u = (phase[i] + tsec * POSE_PLAYBACK * src_fps / span) % 2.0
            k = k0 + int((u if u <= 1.0 else 2.0 - u) * span)
            a = base_adr[i]
            # Re-aim the whole clip onto this slot's facing, then keep the motion's
            # own travel as a local offset so footwork still moves the body instead
            # of the legs sliding under a pinned root.
            local = rot2[i] @ (mo["root_pos"][k, :2] - clip_centre[i])
            d.qpos[a:a + 3] = [pos[i, 0] + local[0], pos[i, 1] + local[1],
                               mo["root_pos"][k, 2]]
            d.qpos[a + 3:a + 7] = qmul_wxyz(yaw_q[i],
                                            quat_xyzw_to_wxyz(mo["root_quat_xyzw"][k]))
            for adr, col in jmap[i].items():
                d.qpos[adr] = mo["joint_pos"][k, col]
        mujoco.mj_forward(m, d)

        dist, az, el = look_at(E[f], T[f])
        cam.lookat[:] = T[f]; cam.distance = dist; cam.azimuth = az; cam.elevation = el
        m.vis.global_.fovy = float(F[f])

        r.update_scene(d, camera=cam)
        rgb = r.render().astype(np.float32)

        if not (args.preview or args.no_post):
            r.enable_depth_rendering()
            r.update_scene(d, camera=cam)
            dep = r.render()
            r.disable_depth_rendering()
            focus = float(np.median(dep[dep < 60.0])) if np.any(dep < 60.0) else dist
            rgb = depth_of_field(rgb, dep, focus)
        if not args.no_post:
            rgb = grade(bloom(rgb))
        full = np.clip(rgb, 0, 255).astype(np.uint8)
        crop_off = max(0, (full.shape[0] - (int(round(full.shape[1] / 2.39)) & ~1)) // 2)
        for a in annos:
            al = anno_alpha(tsec, a)
            if al <= 0.01:
                continue
            ri = a["robot"]
            head = np.array([d.qpos[base_adr[ri]], d.qpos[base_adr[ri] + 1],
                             d.qpos[base_adr[ri] + 2] + 0.62])
            pr = project(head, E[f], T[f], float(F[f]), RW, RH)
            if pr is None:
                continue
            full = draw_hud(full, pr[0], pr[1], a["label"], a["title"], al, crop_off=0)
        frame = letterbox(full)
        if args.ss != 1.0:
            # INTER_AREA is a true box filter: this is the supersample resolve, and
            # it is what removes the stair-stepping MSAA cannot reach.
            tw = args.width & ~1
            th = int(round(frame.shape[0] * tw / frame.shape[1])) & ~1
            frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
        _ = crop_off
        out_shape = frame.shape
        writer.append_data(frame)

        if f % 30 == 0 or f == nframes - 1:
            el_s = time.time() - t0
            print(f"  frame {f+1}/{nframes}  {el_s:6.1f}s elapsed  "
                  f"eta {el_s/(f+1)*(nframes-f-1)/60:5.1f} min", flush=True)

    writer.close()
    meta = {"robots": N_ROBOTS, "clips": clips, "zones": zone,
            "annotations": [{k: a[k] for k in ("robot", "zone", "label", "t0", "t1")}
                            for a in annos],
            "duration_s": nframes / args.fps,
            "fps": args.fps, "resolution": list(out_shape[1::-1]),
            "beats": {"0-5": "establish, all 30", "5-15": "descend + push in",
                      "15-45": "travel + per-robot showcase",
                      "45-55": "pull back", "55-60": "final wide"}}
    with open(os.path.splitext(args.out)[0] + ".json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nwrote {args.out}  ({meta['duration_s']:.1f}s, {meta['resolution']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
