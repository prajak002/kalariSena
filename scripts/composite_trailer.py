#!/usr/bin/env python3
"""Composite the HUD annotations onto a rendered frame sequence and encode.

Cycles renders the world; the annotation layer is 2D and belongs in post, where it
can be restyled without re-rendering nine hours of frames.

Anchors come from the same export the renderer consumed: the per-frame camera and
the per-robot head positions are baked in camera.npz / anim.npz, so a marker lands
on the robot rather than near it, and the pointer cannot drift out of sync with the
camera because both are the same data.

Usage
    python3 scripts/composite_trailer.py --frames frames30 --export export_cycles30 \
        --out results_sequences/army_trailer_cycles.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from scripts.render_army_trailer import anno_alpha, draw_hud, letterbox, project


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--crf", type=int, default=15)
    ap.add_argument("--no-hud", action="store_true")
    ap.add_argument("--full-frame", action="store_true",
                    help="keep the full 16:9 frame instead of the 2.39:1 "
                         "scope crop")
    args = ap.parse_args()

    S = json.load(open(os.path.join(args.export, "scene.json")))
    cam = np.load(os.path.join(args.export, "camera.npz"))
    anim = np.load(os.path.join(args.export, "anim.npz"))
    eye, target, fovy = cam["eye"], cam["target"], cam["fovy"]
    heads = anim["heads"]
    annos = S["annotations"]

    files = sorted(f for f in os.listdir(args.frames) if f.endswith(".png"))
    if not files:
        raise SystemExit(f"no PNG frames in {args.frames}")
    print(f"compositing {len(files)} frames, {len(annos)} annotations")

    import imageio.v2 as imageio
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = imageio.get_writer(
        args.out, fps=args.fps, codec="libx264", macro_block_size=None,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", str(args.crf),
                       "-preset", "slow", "-tune", "film",
                       "-movflags", "+faststart"])

    missing = 0
    for i, fn in enumerate(files):
        img = cv2.imread(os.path.join(args.frames, fn), cv2.IMREAD_COLOR)
        if img is None:
            missing += 1
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        if not args.no_hud and i < len(eye):
            for a in annos:
                al = anno_alpha(i / args.fps, a)
                if al <= 0.01:
                    continue
                pr = project(heads[min(i, len(heads) - 1)][a["robot"]],
                             eye[i], target[i], float(fovy[i]), w, h)
                if pr is None:
                    continue
                img = draw_hud(img, pr[0], pr[1], a["label"], a["title"], al,
                               crop_off=0)
        writer.append_data(img if args.full_frame else letterbox(img))
        if i % 120 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)
    writer.close()
    if missing:
        print(f"[warn] {missing} frame(s) unreadable and skipped")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
