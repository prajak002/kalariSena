#!/usr/bin/env python3
"""Build one continuous showcase video per category from selected G1 motions.

Produces, in --out (default results_sequences/):
    <category>_6moves.mp4     six moves back to back, one continuous take
    <category>_6moves.json    machine-readable timeline: move name, start/end seconds

Continuity is the whole point, so the moves are not stitched as separate clips.
Every move is replayed into the SAME MuJoCo scene, in one pass, under one camera
rig, and consecutive moves are joined by a generated transition rather than a cut:

  * facing is canonicalised   each move is yawed so its front bearing points at the
                              camera, so the performer never spins between moves
  * position is centred       each move is translated so it plays around the origin,
                              instead of chaining end-to-end and drifting metres away
  * joins are interpolated    root position lerps, root orientation slerps and joints
                              lerp across a smoothstep, over a duration scaled by how
                              far the body has to travel, so the join reads as the
                              performer resetting rather than as a jump cut

The camera keeps a constant azimuth, elevation and distance throughout. Only the
look-at point drifts, and it is low-pass filtered over about a second so it reads
as a steady operator holding frame rather than as a tracking shot.

Usage
    python3 scripts/build_sequence_videos.py --selection sel.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.render_compare import (CAMERA_LOOKAT_MIN_HEIGHT, FRONT_CAMERA_AZIMUTH_OFFSET,
                                    FRONT_CAMERA_ELEVATION, front_bearing_deg)
from scripts.render_g1_motion import load_npz_motion
from src.sim.conventions import quat_xyzw_to_wxyz
from src.sim.mujoco_runtime import G1MujocoRuntime, select_gl_backend
from src.sim.rollout import write_video

CORR_DIR = "data/motions_retargeted"
VIDEO_FPS = 30
W, H = 1920, 1440                # showcase renders at a higher resolution than review
SEQ_CAMERA_DISTANCE = 3.6        # wider than the review camera: the body travels here
LOOKAT_SMOOTH_S = 1.0            # seconds of moving average on the look-at point
HOLD_S = 0.5                     # still beat at the very start and end
MIN_BLEND_S, MAX_BLEND_S = 0.45, 1.6
BLEND_SPEED = 1.2                # metres per second the body is allowed to reposition


# ---------------------------------------------------------------- quaternions
def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, xyzw convention."""
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def qslerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    d = float(np.dot(a, b))
    if d < 0.0:                     # take the short way round
        b, d = -b, -d
    if d > 0.9995:
        return (a + t * (b - a)) / np.linalg.norm(a + t * (b - a))
    th = np.arccos(d) * t
    c = b - a * d
    c /= np.linalg.norm(c)
    return a * np.cos(th) + c * np.sin(th)


def yaw_quat(deg: float) -> np.ndarray:
    r = np.radians(deg) / 2.0
    return np.array([0.0, 0.0, np.sin(r), np.cos(r)])


def smoothstep(t: np.ndarray | float):
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------- motion prep
def canonicalise(motion: dict) -> dict:
    """Yaw the move so it faces the camera, and centre it on the origin.

    Without this, each move would start wherever its own reconstruction happened to
    put it and face whichever way that clip was shot, so the performer would appear
    to teleport and spin between moves.
    """
    q = motion["root_quat_xyzw"].copy()
    p = motion["root_pos"].copy()
    delta = -front_bearing_deg(q)          # rotate the mean facing onto +x
    qd = yaw_quat(delta)
    c, s = np.cos(np.radians(delta)), np.sin(np.radians(delta))
    rot = np.array([[c, -s], [s, c]])
    centre = p[:, :2].mean(axis=0)
    p[:, :2] = (p[:, :2] - centre) @ rot.T
    q = np.array([qmul(qd, qi) for qi in q])
    out = dict(motion)
    out["root_pos"], out["root_quat_xyzw"] = p, q
    return out


def blend(a: dict, b: dict, fps: float) -> dict:
    """Generated frames joining the end of `a` to the start of `b`."""
    p0, q0, j0 = a["root_pos"][-1], a["root_quat_xyzw"][-1], a["joint_pos"][-1]
    p1, q1, j1 = b["root_pos"][0], b["root_quat_xyzw"][0], b["joint_pos"][0]
    dist = float(np.linalg.norm(p1[:2] - p0[:2]))
    secs = float(np.clip(MIN_BLEND_S + dist / BLEND_SPEED, MIN_BLEND_S, MAX_BLEND_S))
    n = max(2, int(round(secs * fps)))
    w = smoothstep(np.linspace(0.0, 1.0, n, endpoint=False)[1:])
    return {
        "root_pos": np.array([p0 + (p1 - p0) * t for t in w]),
        "root_quat_xyzw": np.array([qslerp(q0, q1, float(t)) for t in w]),
        "joint_pos": np.array([j0 + (j1 - j0) * t for t in w]),
    }


def hold(m: dict, idx: int, n: int) -> dict:
    # Resolve negative indices explicitly: m[-1:0] is an EMPTY slice, so the closing
    # beat would silently disappear rather than hold the final pose.
    i = idx if idx >= 0 else len(m["root_pos"]) + idx
    n = max(1, n)
    return {"root_pos": np.repeat(m["root_pos"][i:i + 1], n, axis=0),
            "root_quat_xyzw": np.repeat(m["root_quat_xyzw"][i:i + 1], n, axis=0),
            "joint_pos": np.repeat(m["joint_pos"][i:i + 1], n, axis=0)}


def build_track(ids: list[str], names: list[str], fps: float):
    """Concatenate the moves into one pose track plus a segment timeline."""
    moves = []
    ref_joints = None
    for mid in ids:
        m = load_npz_motion(os.path.join(CORR_DIR, f"{mid}.npz"))
        if ref_joints is None:
            ref_joints = list(m["joint_names"])
        elif list(m["joint_names"]) != ref_joints:
            # Reorder rather than trust position: a silently mismatched column would
            # drive the wrong joint and look like a bad retarget.
            idx = [list(m["joint_names"]).index(j) for j in ref_joints]
            m = dict(m); m["joint_pos"] = m["joint_pos"][:, idx]
        moves.append(canonicalise(m))

    stride = max(1, int(round(float(moves[0]["fps"] or 30.0) / fps)))
    for m in moves:
        for k in ("root_pos", "root_quat_xyzw", "joint_pos"):
            m[k] = m[k][::stride]

    P, Q, J, segs = [], [], [], []
    def push(chunk):
        P.append(chunk["root_pos"]); Q.append(chunk["root_quat_xyzw"]); J.append(chunk["joint_pos"])
    def count():
        return sum(len(c) for c in P)

    push(hold(moves[0], 0, int(HOLD_S * fps)))
    for i, m in enumerate(moves):
        start = count()
        push(m)
        segs.append({"index": i + 1, "motion_id": ids[i], "move": names[i],
                     "start_s": round(start / fps, 2), "end_s": round(count() / fps, 2)})
        if i + 1 < len(moves):
            push(blend(m, moves[i + 1], fps))
    push(hold(moves[-1], -1, int(HOLD_S * fps)))
    return (np.concatenate(P), np.concatenate(Q), np.concatenate(J), ref_joints, segs)


def smooth_lookat(root_pos: np.ndarray, fps: float) -> np.ndarray:
    """Low-pass the look-at so the camera glides instead of tracking every step."""
    n = max(1, int(round(LOOKAT_SMOOTH_S * fps)))
    pad = np.pad(root_pos[:, :2], ((n, n), (0, 0)), mode="edge")
    k = np.ones(n) / n
    xy = np.stack([np.convolve(pad[:, i], k, mode="same")[n:-n] for i in range(2)], axis=1)
    z = np.full((len(xy), 1), max(CAMERA_LOOKAT_MIN_HEIGHT, float(root_pos[:, 2].mean())))
    return np.concatenate([xy, z], axis=1)


def render(rt: G1MujocoRuntime, P, Q, J, joint_names, fps):
    mj = rt.mujoco
    idx = {n: i for i, n in enumerate(joint_names)}
    col = np.array([idx.get(j, -1) for j in rt.act_joint_names])
    default_q = rt.default_qpos()
    look = smooth_lookat(P, fps)
    # make_renderer raises the MJCF's 640x480 offscreen buffer to fit, and falls
    # back to whatever the buffer allows if the GL backend refuses the write.
    global W, H
    r = rt.make_renderer(W, H)
    H, W = r.height, r.width
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.distance = SEQ_CAMERA_DISTANCE
    cam.elevation = FRONT_CAMERA_ELEVATION
    cam.azimuth = FRONT_CAMERA_AZIMUTH_OFFSET     # canonicalised facing is +x
    frames = []
    for k in range(len(P)):
        qpos = default_q.copy()
        qpos[0:3] = P[k]
        qpos[3:7] = quat_xyzw_to_wxyz(Q[k])
        for a in range(rt.model.nu):
            if col[a] >= 0:
                qpos[rt.act_qadr[a]] = J[k, col[a]]
        rt.data.qpos[:] = qpos
        rt.data.qvel[:] = 0.0
        mj.mj_forward(rt.model, rt.data)
        cam.lookat[:] = look[k]
        r.update_scene(rt.data, camera=cam)
        frames.append(r.render().copy())
    return frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True,
                    help="JSON: {category: [{id, name, ...}, ...]}")
    ap.add_argument("--out", default="results_sequences")
    ap.add_argument("--only", default=None, help="comma-separated categories")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sel = json.load(open(args.selection))
    cats = ([c.strip() for c in args.only.split(",")] if args.only else list(sel))

    select_gl_backend()
    rt = G1MujocoRuntime()
    for cat in cats:
        items = sel[cat]
        ids = [it["id"] for it in items]
        names = [it["name"] for it in items]
        print(f"\n=== {cat}: {len(ids)} moves ===")
        P, Q, J, jn, segs = build_track(ids, names, VIDEO_FPS)
        frames = render(rt, P, Q, J, jn, VIDEO_FPS)
        stem = os.path.join(args.out, f"{cat.lower()}_6moves")
        write_video(f"{stem}.mp4", frames, fps=VIDEO_FPS)
        meta = {"category": cat, "fps": VIDEO_FPS, "resolution": [W, H],
                "duration_s": round(len(frames) / VIDEO_FPS, 2),
                "camera": {"azimuth_deg": FRONT_CAMERA_AZIMUTH_OFFSET,
                           "elevation_deg": FRONT_CAMERA_ELEVATION,
                           "distance_m": SEQ_CAMERA_DISTANCE,
                           "lookat": "smoothed root position, constant angle"},
                "segments": segs}
        with open(f"{stem}.json", "w") as fh:
            json.dump(meta, fh, indent=2)
        for s in segs:
            print(f"   {s['index']}. {s['move']:<34s} {s['start_s']:>6.2f} - {s['end_s']:>6.2f}s")
        print(f"   wrote {stem}.mp4 ({meta['duration_s']}s) and {stem}.json")
    rt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
