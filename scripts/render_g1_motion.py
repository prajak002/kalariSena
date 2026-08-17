#!/usr/bin/env python3
"""Render a retargeted G1 motion to mp4 for side-by-side human/robot review.

Input contract is the NPZ written by scripts/annotate_motion_library.py:

    q               [T, nq]     full configuration (root + joints)
    dq              [T, nv]
    root_pos        [T, 3]
    root_quat_xyzw  [T, 4]      NOTE: Pinocchio order (x, y, z, w)
    joint_pos       [T, n_joints]
    joint_cols      [n_joints]  joint names, bytes
    contacts        [T, 2]
    phase           [T]
    family, fps, motion_id

This is KINEMATIC PLAYBACK: qpos is written directly per frame and mj_forward is
called for rendering only. No torques, no contact response, no physics
integration. It shows what the retarget produced, which is exactly what a visual
retarget review needs. It is not a claim that the robot can execute the motion.

Two conventions are handled here and both are load-bearing:
  * root_quat_xyzw is Pinocchio order; MuJoCo qpos wants [w, x, y, z].
  * joint columns are matched to MuJoCo actuators BY NAME, never by index order.

Usage
    python3 scripts/render_g1_motion.py --npz data/motions_retargeted/foo.npz
    python3 scripts/render_g1_motion.py --npz-dir data/motions_retargeted --out results_review
    python3 scripts/render_g1_motion.py --authored --out results_review   # demo without real data
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sim.conventions import quat_xyzw_to_wxyz
from src.sim.mujoco_runtime import DEFAULT_STANDING_POSE, G1MujocoRuntime, select_gl_backend
from src.sim.rollout import write_video

LABEL = "kinematic playback (no physics)"


def _s(value) -> str:
    """Decode a numpy bytes/str scalar."""
    arr = np.asarray(value).reshape(-1)
    v = arr[0]
    return v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v)


def load_npz_motion(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    files = set(data.files)

    if "joint_pos" in files:
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
    elif "q" in files:
        joint_pos = np.asarray(data["q"], dtype=np.float64)[:, 7:]
    else:
        raise KeyError(f"{path}: needs 'joint_pos' or 'q'")

    n = joint_pos.shape[0]

    if "joint_cols" in files:
        # GEM-X CSV headers carry a "_dof" suffix ("left_knee_joint_dof") which
        # annotate_motion_library preserves. Strip it so name-keyed matching
        # against MuJoCo actuator names ("left_knee_joint") works; without this
        # every joint silently fails to match and the replay is a frozen pose.
        names = [
            (c.decode("utf-8") if isinstance(c, (bytes, np.bytes_)) else str(c))
            .removesuffix("_dof")
            for c in np.asarray(data["joint_cols"]).reshape(-1)
        ]
    else:
        names = None

    if "root_pos" in files:
        root_pos = np.asarray(data["root_pos"], dtype=np.float64).reshape(n, 3)
    elif "q" in files:
        root_pos = np.asarray(data["q"], dtype=np.float64)[:, 0:3]
    else:
        root_pos = np.zeros((n, 3))
        root_pos[:, 2] = 0.79

    if "root_quat_xyzw" in files:
        quat_xyzw = np.asarray(data["root_quat_xyzw"], dtype=np.float64).reshape(n, 4)
    elif "q" in files:
        quat_xyzw = np.asarray(data["q"], dtype=np.float64)[:, 3:7]
    else:
        quat_xyzw = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))

    return {
        "motion_id": _s(data["motion_id"]) if "motion_id" in files
        else os.path.splitext(os.path.basename(path))[0],
        "family": _s(data["family"]) if "family" in files else "unlabeled",
        "fps": float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in files else 30.0,
        "joint_pos": joint_pos,
        "joint_names": names,
        "root_pos": root_pos,
        "root_quat_xyzw": quat_xyzw,
        "contacts": np.asarray(data["contacts"]).astype(bool) if "contacts" in files else None,
        "source": "retargeted_npz",
        "npz_path": path,
    }


def load_authored_motion(motion_id: str, rt: G1MujocoRuntime) -> dict:
    """Stand-in so the render + review pipeline is demonstrable without real data."""
    from src.sim.motions import get_motion

    base = {n: 0.0 for n in rt.act_joint_names}
    base.update(DEFAULT_STANDING_POSE)
    m = get_motion(motion_id, rt.act_joint_names, base, fps=rt.CONTROL_HZ)
    n = m.n_frames
    return {
        "motion_id": motion_id,
        "family": m.family,
        "fps": m.fps,
        "joint_pos": m.q_ref,
        "joint_names": list(m.joint_names),
        "root_pos": None,           # solved per frame
        "root_quat_xyzw": np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)),
        "contacts": m.contacts_ref,
        "source": "authored_keyframe",
        "npz_path": None,
    }


def render_motion(motion: dict, rt: G1MujocoRuntime, out_dir: str,
                  camera: str = "track", video_fps: int = 30,
                  ground_clamp: bool = False, width: int = 640, height: int = 480) -> dict:
    """Render one motion; returns a diagnostics record for the review page."""
    mujoco = rt.mujoco
    joint_pos = motion["joint_pos"]
    n_frames = joint_pos.shape[0]
    src_fps = float(motion["fps"]) or 30.0

    # --- joint mapping BY NAME -------------------------------------------
    names = motion["joint_names"]
    if names is None:
        if joint_pos.shape[1] != rt.model.nu:
            raise ValueError(
                f"{motion['motion_id']}: {joint_pos.shape[1]} joint columns and no "
                f"joint_cols to match against {rt.model.nu} actuators"
            )
        col_for_act = np.arange(rt.model.nu)
        matched, missing, extra = list(rt.act_joint_names), [], []
    else:
        index = {n: i for i, n in enumerate(names)}
        col_for_act = np.full(rt.model.nu, -1, dtype=int)
        matched, missing = [], []
        for a, jn in enumerate(rt.act_joint_names):
            if jn in index:
                col_for_act[a] = index[jn]
                matched.append(jn)
            else:
                missing.append(jn)
        extra = [n for n in names if n not in set(rt.act_joint_names)]

    foot_geoms = rt.left_foot_geoms + rt.right_foot_geoms
    default_q = rt.default_qpos()

    frame_stride = max(1, int(round(src_fps / video_fps)))
    frames = []
    lowest_per_frame = []

    for k in range(n_frames):
        qpos = default_q.copy()

        if motion["root_pos"] is not None:
            qpos[0:3] = motion["root_pos"][k]
        qpos[3:7] = quat_xyzw_to_wxyz(motion["root_quat_xyzw"][k])

        for a in range(rt.model.nu):
            c = col_for_act[a]
            if c >= 0:
                qpos[rt.act_qadr[a]] = joint_pos[k, c]

        rt.data.qpos[:] = qpos
        rt.data.qvel[:] = 0.0
        mujoco.mj_forward(rt.model, rt.data)

        lowest = min(
            float(rt.data.geom_xpos[g][2]) - float(rt.model.geom_size[g][0])
            for g in foot_geoms
        )
        # Only *record* this as a retarget defect when the source actually carries
        # a root track. Otherwise the base height is ours, not the retarget's, and
        # reporting it as ground penetration or float would be wrong.
        if motion["root_pos"] is not None:
            lowest_per_frame.append(lowest)

        if ground_clamp or motion["root_pos"] is None:
            qpos[2] += 0.001 - lowest
            rt.data.qpos[:] = qpos
            mujoco.mj_forward(rt.model, rt.data)

        if k % frame_stride == 0:
            frames.append(rt.render_frame(camera=camera, width=width, height=height))

    out_path = os.path.join(out_dir, f"robot_{motion['motion_id']}.mp4")
    ok = write_video(out_path, frames, fps=video_fps)

    has_root_track = len(lowest_per_frame) > 0
    lowest_arr = np.array(lowest_per_frame) if has_root_track else None
    rec = {
        "motion_id": motion["motion_id"],
        "family": motion["family"],
        "source": motion["source"],
        "npz_path": motion["npz_path"],
        "robot_video": out_path if ok else None,
        "frames": int(n_frames),
        "fps": src_fps,
        "duration_s": round(n_frames / src_fps, 3),
        "render_label": LABEL,
        "ground_clamped": bool(ground_clamp or motion["root_pos"] is None),
        # --- retarget quality flags, all measured -------------------------
        "joints_matched": len(matched),
        "joints_expected": int(rt.model.nu),
        "joints_missing": missing,
        "joints_unused_in_source": extra,
        "has_root_track": has_root_track,
        "foot_penetration_max_m": (
            round(float(-lowest_arr.min()), 4) if has_root_track and lowest_arr.min() < 0 else 0.0),
        "foot_float_max_m": (
            round(float(lowest_arr.max()), 4) if has_root_track and lowest_arr.max() > 0 else 0.0),
        "frames_penetrating": int((lowest_arr < -0.01).sum()) if has_root_track else 0,
        "frames_floating": int((lowest_arr > 0.05).sum()) if has_root_track else 0,
    }
    flags = []
    if rec["joints_missing"]:
        flags.append(f"{len(rec['joints_missing'])} joints missing from source")
    if not has_root_track:
        flags.append("no root track in source (base height synthesised)")
    else:
        if rec["frames_penetrating"] > 0.05 * n_frames:
            flags.append(f"ground penetration in {rec['frames_penetrating']}/{n_frames} frames "
                         f"(max {rec['foot_penetration_max_m']} m)")
        if rec["frames_floating"] > 0.05 * n_frames:
            flags.append(f"floating in {rec['frames_floating']}/{n_frames} frames "
                         f"(max {rec['foot_float_max_m']} m)")
    rec["quality_flags"] = flags

    print(f"  {rec['motion_id']:34s} {rec['frames']:4d}f {rec['duration_s']:6.2f}s  "
          f"joints {rec['joints_matched']}/{rec['joints_expected']}  "
          f"{'FLAGS: ' + '; '.join(flags) if flags else 'clean'}")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=None, help="single retargeted NPZ")
    ap.add_argument("--npz-dir", default=None, help="directory of retargeted NPZs")
    ap.add_argument("--authored", action="store_true",
                    help="render the 3 authored stand-in motions (no real data needed)")
    ap.add_argument("--out", default="results_review")
    ap.add_argument("--camera", default="track", choices=["track", "front", "side"])
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--ground-clamp", action="store_true",
                    help="shift base so the lowest foot sphere rests on the floor")
    args = ap.parse_args()

    if not (args.npz or args.npz_dir or args.authored):
        ap.error("pass --npz, --npz-dir or --authored")

    os.makedirs(args.out, exist_ok=True)
    select_gl_backend()
    rt = G1MujocoRuntime()
    print(f"*** {LABEL} *** shows the retarget, not executable robot behaviour")

    motions = []
    if args.npz:
        motions.append(load_npz_motion(args.npz))
    if args.npz_dir:
        if not os.path.isdir(args.npz_dir):
            raise SystemExit(f"not a directory: {args.npz_dir}")
        files = sorted(f for f in os.listdir(args.npz_dir) if f.endswith(".npz"))
        if not files:
            raise SystemExit(f"no .npz files in {args.npz_dir}")
        print(f"found {len(files)} NPZ files in {args.npz_dir}")
        motions += [load_npz_motion(os.path.join(args.npz_dir, f)) for f in files]
    if args.authored:
        from src.sim.motions import MOTION_IDS
        motions += [load_authored_motion(m, rt) for m in MOTION_IDS]

    records = [
        render_motion(m, rt, args.out, camera=args.camera, video_fps=args.fps,
                      ground_clamp=args.ground_clamp)
        for m in motions
    ]

    manifest = os.path.join(args.out, "render_manifest.json")
    with open(manifest, "w") as fh:
        json.dump(records, fh, indent=2)
    print(f"\nrendered {len(records)} motion(s) -> {args.out}")
    print(f"wrote {manifest}")
    flagged = [r for r in records if r["quality_flags"]]
    print(f"clean: {len(records) - len(flagged)}   flagged for review: {len(flagged)}")
    rt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
