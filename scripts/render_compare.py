#!/usr/bin/env python3
"""DRDO-review renders: raw vs corrected G1 motion, same scene, same fixed camera.

For every motion that has BOTH a raw NPZ (data/motions_retargeted_raw/) and a
corrected NPZ (data/motions_retargeted/), renders in the SAME MuJoCo scene with
an IDENTICAL fixed front-view camera (so the two videos are frame-aligned and
can be blended or crossfaded):

  results_review/mjc_raw_<id>.mp4     raw automatic generation, MuJoCo render
  results_review/mjc_<id>.mp4         Pragya-corrected motion, MuJoCo render
  results_review/overlay_<id>.mp4     both superimposed: corrected solid,
                                      raw as a red-tinted ghost
  results_review/img_raw_<id>.jpg     still image, raw (mid-motion frame)
  results_review/img_corr_<id>.jpg    still image, corrected (same frame)

Camera: fixed (NOT tracking) front view, centred on the corrected motion's mean
root position. Fixed is required because a tracking camera would follow each
version's own CoM and the frames would no longer align.

Writes results_review/compare_manifest.json for the review page.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.render_g1_motion import load_npz_motion
from src.sim.conventions import quat_xyzw_to_wxyz
from src.sim.mujoco_runtime import G1MujocoRuntime, select_gl_backend
from src.sim.rollout import write_video

RAW_DIR = "data/motions_retargeted_raw"
CORR_DIR = "data/motions_retargeted"
OUT = "results_review"
GHOST_ALPHA = 0.42
VIDEO_FPS = 30
# These three clips (mjc_raw_/mjc_/overlay_) are what the gallery shows side by
# side, and the panes are up to 760 CSS px wide -- ~1520 device px on a retina
# display. Rendering at 640x480 meant every one was upscaled 2.4x in CSS, which
# is what read as pixelation. make_renderer raises the MJCF's matching 640x480
# offscreen buffer to fit, so the request is no longer silently clamped.
W, H = G1MujocoRuntime.RENDER_W, G1MujocoRuntime.RENDER_H

# ---------------------------------------------------------------------------
# Front-view camera.
#
# The G1's canonical forward axis is pelvis +x. That is read off the model, not
# assumed: each foot carries contact spheres at local x = +0.12 (toe) and
# x = -0.05 (heel), so +x points out of the toes.
#
# MuJoCo's free camera is placed at  lookat - distance * forward(azimuth, elevation)
# and looks ALONG +forward. Azimuth therefore names the direction the camera
# LOOKS, not the bearing it stands on. Pointing azimuth along the robot's own
# heading puts the camera behind it and renders the back, which is the bug this
# offset fixes: to see the FRONT the camera must look back down the forward axis,
# i.e. heading + 180.
#
# This rotates only the camera. The retargeted motion, joint angles and the
# robot's anatomical left/right are untouched, so nothing is mirrored.
FRONT_CAMERA_AZIMUTH_OFFSET = 180.0   # deg, added to the motion's mean heading
FRONT_CAMERA_ELEVATION = -12.0        # deg, slight downward tilt
FRONT_CAMERA_DISTANCE = 2.9           # metres from the look-at point
CAMERA_LOOKAT_MIN_HEIGHT = 0.55       # metres, keeps the torso framed in low stances


def camera_lookat(root_pos: np.ndarray) -> np.ndarray:
    """Look-at point: the motion's mean root position, floored to torso height.

    Deep Kalari stances drag the mean root down far enough that a raw mean would
    aim the camera at the floor, so the height is clamped.
    """
    lookat = root_pos.mean(axis=0).copy()
    lookat[2] = max(CAMERA_LOOKAT_MIN_HEIGHT, float(root_pos[:, 2].mean()))
    return lookat


def front_camera(mujoco, lookat: np.ndarray, heading_deg: float):
    """Fixed camera framing the robot's FRONT, centred on its torso."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = FRONT_CAMERA_DISTANCE
    cam.azimuth = heading_deg + FRONT_CAMERA_AZIMUTH_OFFSET
    cam.elevation = FRONT_CAMERA_ELEVATION
    return cam


def mean_heading_deg(quats_xyzw: np.ndarray) -> float:
    """Mean yaw heading of the root, from the body +x (forward) axis."""
    from src.sim.conventions import quat_wxyz_to_matrix, quat_xyzw_to_wxyz as _x2w
    fx, fy = 0.0, 0.0
    for q in quats_xyzw[:: max(1, len(quats_xyzw) // 40)]:
        f = quat_wxyz_to_matrix(_x2w(q))[:, 0]   # body forward in world
        fx += f[0]; fy += f[1]
    return float(np.degrees(np.arctan2(fy, fx)))


def front_bearing_deg(quats_xyzw: np.ndarray) -> float:
    """Bearing from the robot toward the camera that shows its front most often.

    This is the weighted circular mean of the body's forward axis, which is the
    bearing maximising the average frontality  sum(w_i * cos(theta_i))  over the
    clip. Two alternatives were measured across all 70 clips and both lost:

        circular mean (this)   mean cos 0.69,  79.1% of frames within 60deg of front
        max count of cos>0     mean cos 0.46,  51.4%
        max sum of cos^2       mean cos 0.67,  78.9%

    Maximising the *count* of frames with cos>0 looks appealing and is wrong: a
    frame 89 degrees off still scores as "not facing away" while rendering the
    performer side-on, so the optimiser happily parks the camera on the flank.
    Judge a camera bearing by how square the body is to it, never by how many
    frames merely avoid being back-on.

    Frames are weighted by how horizontal the forward axis is: in deep squats and
    forward bends the pelvis pitches until its +x points nearly straight down and
    the yaw derived from it is numerically meaningless, so those frames must not
    be allowed to choose the camera.
    """
    from src.sim.conventions import quat_wxyz_to_matrix, quat_xyzw_to_wxyz as _x2w
    fwd = np.array([quat_wxyz_to_matrix(_x2w(q))[:, 0] for q in quats_xyzw])
    xy = fwd[:, :2]
    horiz = np.linalg.norm(xy, axis=1)
    if not np.any(horiz > 1e-6):
        return 0.0
    # summing the raw (unnormalised) vectors already weights each frame by how
    # horizontal it is, which is exactly the weighting we want
    v = xy.sum(axis=0)
    return float(np.degrees(np.arctan2(v[1], v[0])))


def pose_frames(rt: G1MujocoRuntime, motion: dict, cam, stride: int) -> list[np.ndarray]:
    """Kinematic replay frames with an explicit camera object."""
    mujoco = rt.mujoco
    idx = {n: i for i, n in enumerate(motion["joint_names"] or [])}
    col_for_act = np.array([idx.get(j, -1) for j in rt.act_joint_names])
    default_q = rt.default_qpos()
    r = rt.make_renderer(W, H)
    frames = []
    n = motion["joint_pos"].shape[0]
    for k in range(0, n, stride):
        qpos = default_q.copy()
        qpos[0:3] = motion["root_pos"][k]
        qpos[3:7] = quat_xyzw_to_wxyz(motion["root_quat_xyzw"][k])
        for a in range(rt.model.nu):
            if col_for_act[a] >= 0:
                qpos[rt.act_qadr[a]] = motion["joint_pos"][k, col_for_act[a]]
        rt.data.qpos[:] = qpos
        rt.data.qvel[:] = 0.0
        mujoco.mj_forward(rt.model, rt.data)
        r.update_scene(rt.data, camera=cam)
        frames.append(r.render().copy())
    return frames


def background_frame(rt: G1MujocoRuntime, cam) -> np.ndarray:
    """One robot-free frame: teleport the robot far below the floor and render."""
    qpos = rt.default_qpos()
    qpos[2] = -50.0
    rt.data.qpos[:] = qpos
    rt.mujoco.mj_forward(rt.model, rt.data)
    r = rt.make_renderer(W, H)
    r.update_scene(rt.data, camera=cam)
    return r.render().copy()


def ghost_blend(corr: np.ndarray, raw: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Corrected solid + raw as red ghost, masked to the raw robot's pixels only.

    A global alpha blend tints the whole background; instead the raw robot's
    silhouette is isolated by differencing against a robot-free render of the
    same scene, and only those pixels receive the tinted ghost.
    """
    diff = np.abs(raw.astype(np.int16) - bg.astype(np.int16)).sum(axis=2)
    mask = diff > 28   # raw-robot silhouette (incl. its shadow edge)
    tinted = raw.astype(np.float32)
    tinted[..., 0] = np.clip(tinted[..., 0] * 1.25 + 55, 0, 255)
    tinted[..., 1] *= 0.55
    tinted[..., 2] *= 0.55
    out = corr.astype(np.float32)
    out[mask] = (1 - GHOST_ALPHA) * out[mask] + GHOST_ALPHA * tinted[mask]
    return np.clip(out, 0, 255).astype(np.uint8)


def save_jpg(path: str, frame: np.ndarray) -> None:
    import imageio.v2 as imageio

    imageio.imwrite(path, frame, quality=88)


def render_one(mid: str, rt: G1MujocoRuntime, force: bool = False) -> dict | None:
    raw_p = os.path.join(RAW_DIR, f"{mid}.npz")
    corr_p = os.path.join(CORR_DIR, f"{mid}.npz")
    if not (os.path.isfile(raw_p) and os.path.isfile(corr_p)):
        return None
    done = not force and all(os.path.isfile(os.path.join(OUT, f"{pfx}{mid}{ext}"))
               for pfx, ext in (("mjc_raw_", ".mp4"), ("mjc_", ".mp4"),
                                ("overlay_", ".mp4"), ("img_raw_", ".jpg"),
                                ("img_corr_", ".jpg")))
    if done and os.path.getmtime(os.path.join(OUT, f"overlay_{mid}.mp4")) > os.path.getmtime(corr_p):
        print(f"  {mid:26s} cached")
        return {"motion_id": mid, "frames": None, "camera": "fixed front view",
                **{k: os.path.join(OUT, v.format(mid=mid)) for k, v in {
                    "raw_video": "mjc_raw_{mid}.mp4", "corr_video": "mjc_{mid}.mp4",
                    "overlay_video": "overlay_{mid}.mp4", "raw_image": "img_raw_{mid}.jpg",
                    "corr_image": "img_corr_{mid}.jpg"}.items()}}
    raw = load_npz_motion(raw_p)
    corr = load_npz_motion(corr_p)
    fps = float(corr["fps"]) or 30.0
    stride = max(1, int(round(fps / VIDEO_FPS)))

    # One camera, built from the CORRECTED motion, reused for the raw render and
    # the overlay. Both panels must share it or the two videos stop being
    # frame-aligned and the ghost blend is meaningless.
    cam = front_camera(rt.mujoco, camera_lookat(corr["root_pos"]),
                       front_bearing_deg(corr["root_quat_xyzw"]))

    fr_corr = pose_frames(rt, corr, cam, stride)
    fr_raw = pose_frames(rt, raw, cam, stride)
    bg = background_frame(rt, cam)
    n = min(len(fr_corr), len(fr_raw))
    fr_overlay = [ghost_blend(fr_corr[i], fr_raw[i], bg) for i in range(n)]

    paths = {
        "raw_video": os.path.join(OUT, f"mjc_raw_{mid}.mp4"),
        "corr_video": os.path.join(OUT, f"mjc_{mid}.mp4"),
        "overlay_video": os.path.join(OUT, f"overlay_{mid}.mp4"),
        "raw_image": os.path.join(OUT, f"img_raw_{mid}.jpg"),
        "corr_image": os.path.join(OUT, f"img_corr_{mid}.jpg"),
    }
    write_video(paths["raw_video"], fr_raw, fps=VIDEO_FPS)
    write_video(paths["corr_video"], fr_corr, fps=VIDEO_FPS)
    write_video(paths["overlay_video"], fr_overlay, fps=VIDEO_FPS)
    mid_i = n // 2
    save_jpg(paths["raw_image"], fr_raw[mid_i])
    save_jpg(paths["corr_image"], fr_corr[mid_i])

    print(f"  {mid:26s} {n} frames -> raw/corr/overlay videos + stills")
    return {"motion_id": mid, **paths, "frames": n, "camera": "fixed front view"}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="re-render even when outputs look up to date; required "
                         "after a camera change, since the cache only compares "
                         "output mtimes against the source NPZ")
    ap.add_argument("--only", default=None, help="comma-separated motion ids")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    select_gl_backend()
    rt = G1MujocoRuntime()
    mids = sorted(f[:-4] for f in os.listdir(CORR_DIR) if f.endswith(".npz"))
    if args.only:
        keep = {m.strip() for m in args.only.split(",") if m.strip()}
        mids = [m for m in mids if m in keep]
    records, skipped = [], []
    for mid in mids:
        rec = render_one(mid, rt, force=args.force)
        if rec:
            records.append(rec)
        else:
            skipped.append(mid)
    # Merge rather than overwrite: with --only, a plain dump would silently drop
    # every motion this run did not touch and the review page would lose them.
    man_path = os.path.join(OUT, "compare_manifest.json")
    merged = {}
    if os.path.isfile(man_path):
        with open(man_path) as fh:
            for old in json.load(fh):
                merged[old["motion_id"]] = old
    for rec in records:
        merged[rec["motion_id"]] = rec
    with open(man_path, "w") as fh:
        json.dump([merged[k] for k in sorted(merged)], fh, indent=2)
    print(f"\n{len(records)} rendered, {len(skipped)} skipped (missing raw or corrected NPZ)")
    print(f"wrote {OUT}/compare_manifest.json")
    rt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
