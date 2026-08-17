#!/usr/bin/env python3
"""Bake the posed army scene out of MuJoCo into a Blender/Cycles-ready package.

MuJoCo's rasteriser cannot produce publication-grade frames: no global illumination,
no ray-traced reflection, no PBR. Rather than rebuild the scene by hand in Blender
and risk it drifting from the simulation, this exports exactly what MuJoCo computed
and lets Cycles shade it.

Reuses the existing pipeline wholesale: the same formation, the same per-zone clip
selection, the same held-stance playback and the same camera spline that
render_army_trailer.py drives, so the Cycles version is the same film with a better
renderer rather than a different film.

Writes into --out:
    scene.json      static description: meshes, geom->mesh map, floor, zone lines,
                    banner planes with texture paths, annotation schedule
    anim.npz        per-frame world transforms for every mesh geom (F, G, 3/9)
    camera.npz      per-frame eye, target, fovy

Usage
    python3 scripts/export_cycles_scene.py --duration 15 --fps 24 --out export_cycles
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco

from scripts.render_army_trailer import (MOTION_DIR, N_ROBOTS, POSE_PLAYBACK,
                                         ZONE_ORDER, camera_track, clip_heading_deg,
                                         formation, hold_window, pick_zone_clips,
                                         qmul_wxyz)
from scripts.render_g1_motion import load_npz_motion
from src.sim.conventions import quat_xyzw_to_wxyz
from src.sim.mujoco_runtime import select_gl_backend

SCENE = "assets/unitree_g1/army_30.xml"


def mat_to_quat(m: np.ndarray) -> list[float]:
    """3x3 rotation -> wxyz quaternion (Blender wants quaternions, not matrices)."""
    t = float(np.trace(m))
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x = 0.25 * s, (m[2, 1] - m[1, 2]) / s
        y, z = (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x = (m[2, 1] - m[1, 2]) / s, 0.25 * s
        y, z = (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s
        y, z = 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s
        y, z = (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return [float(w), float(x), float(y), float(z)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=SCENE)
    ap.add_argument("--out", default="export_cycles")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--start", type=float, default=0.0,
                    help="export only a window of the film, for a hero test render")
    ap.add_argument("--heroes", type=int, default=3,
                    help="showcased humanoids per zone")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
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

    base_adr, jmap, yaw_q, rot2, clip_centre, windows = [], [], [], [], [], []
    for i in range(N_ROBOTS):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                                f"r{i:02d}_floating_base_joint")
        base_adr.append(m.jnt_qposadr[jid])
        cols = {}
        for k, jn in enumerate(list(motions[i]["joint_names"] or [])):
            j2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"r{i:02d}_{jn}")
            if j2 >= 0:
                cols[m.jnt_qposadr[j2]] = k
        jmap.append(cols)
        dy = np.radians(yaw[i] - clip_heading_deg(motions[i]))
        yaw_q.append(np.array([np.cos(dy / 2), 0.0, 0.0, np.sin(dy / 2)]))
        c, s = np.cos(dy), np.sin(dy)
        rot2.append(np.array([[c, -s], [s, c]]))
        w = hold_window(motions[i])
        windows.append(w)
        clip_centre.append(motions[i]["root_pos"][w[0]:w[1], :2].mean(axis=0))

    # Compress the whole beat structure into --duration rather than truncating it,
    # so a short cut still ends on the finale instead of stopping mid-tour.
    E, T, F, annos = camera_track(pos, zone, args.duration, args.fps,
                                  heroes_per_zone=args.heroes)
    # Overwrite the generic label pool with the hero's ACTUAL clip identity from
    # the motion library: name, library pose id, zone. The annotation may only
    # claim what the robot is really performing.
    library = sorted(f[:-4] for f in os.listdir(MOTION_DIR) if f.endswith(".npz"))
    for a in annos:
        cid = clips[a["robot"]]
        a["label"] = cid.split("_", 1)[-1].replace("_", " ").upper()
        a["title"] = (f"POSE {library.index(cid) + 1:02d}/{len(library)} · "
                      f"{a['zone'].upper()}")
    f0 = int(round(args.start * args.fps))
    f1 = min(len(E), f0 + int(round(args.duration * args.fps)))
    frames = range(f0, f1)
    print(f"exporting frames {f0}..{f1} of {len(E)} ({(f1-f0)/args.fps:.1f}s @ {args.fps}fps)")

    # ---- static description ------------------------------------------------
    # Export the geometry MuJoCo actually compiled, not the source STLs. The
    # compiler re-centres mesh assets on their centre of mass, so raw STL vertices
    # sit at a per-mesh offset from geom_xpos; feeding Blender the STLs produced a
    # formation of exploded robots with limbs scattered around each position.
    mesh_geoms, meshes = [], {}
    mesh_arrays = {}
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = int(m.geom_dataid[g])
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mid)
        if nm not in mesh_arrays:
            # deliberately not f0/f1: those are the frame-range bounds above and
            # shadowing them silently truncated the exported camera track
            va, nv = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            fa, nf = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
            mesh_arrays[f"v_{nm}"] = np.array(m.mesh_vert[va:va + nv],
                                              dtype=np.float32).reshape(-1, 3)
            mesh_arrays[f"f_{nm}"] = np.array(m.mesh_face[fa:fa + nf],
                                              dtype=np.int32).reshape(-1, 3)
        meshes[nm] = "compiled"
        mesh_geoms.append({"geom": int(g), "mesh": nm,
                           "rgba": [float(x) for x in m.geom_rgba[g]]})

    statics = []
    for g in range(m.ngeom):
        gt = m.geom_type[g]
        if gt == mujoco.mjtGeom.mjGEOM_MESH:
            continue
        gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        matid = int(m.geom_matid[g])
        matname = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, matid)
                   if matid >= 0 else "")
        statics.append({
            "name": gname, "kind": int(gt), "material": matname,
            "pos": [float(x) for x in m.geom_pos[g]],
            "quat": [float(x) for x in m.geom_quat[g]],
            "size": [float(x) for x in m.geom_size[g]],
            "rgba": [float(x) for x in m.geom_rgba[g]],
        })

    scene = {
        "fps": args.fps, "n_frames": len(frames), "frame_offset": f0,
        "robots": N_ROBOTS, "clips": clips, "zones": zone,
        "meshdir": "assets/unitree_g1", "meshes": meshes,
        "mesh_geoms": mesh_geoms, "statics": statics,
        "banners": {"kalarisena_hero": "banners/kalarisena_hero.png",
                    "pragya_velvet": "banners/pragya_velvet.png"},
        "annotations": [{**a, "t0": float(a["t0"]), "t1": float(a["t1"])}
                        for a in annos],
        "robot_root_body": [f"r{i:02d}_pelvis" for i in range(N_ROBOTS)],
    }
    with open(os.path.join(args.out, "scene.json"), "w") as fh:
        json.dump(scene, fh, indent=1)

    # ---- per-frame transforms ----------------------------------------------
    G = len(mesh_geoms)
    gidx = np.array([mg["geom"] for mg in mesh_geoms])
    P = np.zeros((len(frames), G, 3), np.float32)
    Q = np.zeros((len(frames), G, 4), np.float32)
    heads = np.zeros((len(frames), N_ROBOTS, 3), np.float32)

    for out_i, f in enumerate(frames):
        tsec = f / args.fps
        for i in range(N_ROBOTS):
            mo = motions[i]
            k0, k1 = windows[i]
            span = max(1, k1 - k0 - 1)
            src_fps = float(mo["fps"] or 30.0)
            u = ((np.random.default_rng(3).uniform(0, 1, N_ROBOTS)[i]
                  + tsec * POSE_PLAYBACK * src_fps / span) % 2.0)
            k = k0 + int((u if u <= 1.0 else 2.0 - u) * span)
            a = base_adr[i]
            local = rot2[i] @ (mo["root_pos"][k, :2] - clip_centre[i])
            d.qpos[a:a + 3] = [pos[i, 0] + local[0], pos[i, 1] + local[1],
                               mo["root_pos"][k, 2]]
            d.qpos[a + 3:a + 7] = qmul_wxyz(yaw_q[i],
                                            quat_xyzw_to_wxyz(mo["root_quat_xyzw"][k]))
            for adr, col in jmap[i].items():
                d.qpos[adr] = mo["joint_pos"][k, col]
        mujoco.mj_forward(m, d)
        P[out_i] = d.geom_xpos[gidx]
        for j, g in enumerate(gidx):
            Q[out_i, j] = mat_to_quat(d.geom_xmat[g].reshape(3, 3))
        for i in range(N_ROBOTS):
            heads[out_i, i] = [d.qpos[base_adr[i]], d.qpos[base_adr[i] + 1],
                               d.qpos[base_adr[i] + 2] + 0.62]
        if out_i % 60 == 0:
            print(f"  baked {out_i+1}/{len(frames)}", flush=True)

    np.savez_compressed(os.path.join(args.out, "meshes.npz"), **mesh_arrays)
    np.savez_compressed(os.path.join(args.out, "anim.npz"), pos=P, quat=Q, heads=heads)
    np.savez_compressed(os.path.join(args.out, "camera.npz"),
                        eye=E[f0:f1].astype(np.float32),
                        target=T[f0:f1].astype(np.float32),
                        fovy=F[f0:f1].astype(np.float32))
    mb = sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out)) / 1e6
    print(f"\nwrote {args.out}/  ({G} mesh instances, {len(frames)} frames, {mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
