#!/usr/bin/env python3
"""Assemble the review page + videos into a static site ready for Vercel.

Produces:
    web/
      index.html          the side-by-side review page
      robot_*.mp4         generated G1 motions
      human/*.mp4         original human videos (copied in, so the link is self-contained)
      vercel.json         static config + video caching headers

Then deploy with:
    npx vercel deploy --prod web

Vercel limits worth knowing before you push 70 motions:
  * 100 MB per file
  * ~100 MB per deployment on Hobby unless the files are served from the CDN
    after upload; large video sets are the usual failure mode.
This script reports total size and flags anything oversized rather than letting
the deploy fail halfway.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess


def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def copy_video(src: str, dst: str, compress: bool) -> None:
    """Copy a video into the site, optionally recompressed for deploy size.

    Compression is deploy-only (sources untouched): scale to <=480p height,
    H.264 CRF 30, faststart. Skips work when dst is already newer than src.
    """
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return
    if not compress:
        shutil.copy2(src, dst)
        return
    r = subprocess.run(
        [_ffmpeg(), "-y", "-i", src, "-vf", "scale=-2:'min(480,ih)'",
         "-c:v", "libx264", "-crf", "30", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst],
        capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(dst):
        shutil.copy2(src, dst)

MAX_FILE_MB = 100.0
WARN_TOTAL_MB = 100.0


def human_size(n_bytes: int) -> str:
    mb = n_bytes / 1e6
    return f"{mb:.1f} MB" if mb < 1000 else f"{mb / 1000:.2f} GB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review-dir", default="results_review",
                    help="directory holding review.html and robot_*.mp4")
    ap.add_argument("--human-dir", default="data/kalari_videos")
    ap.add_argument("--out", default="web")
    ap.add_argument("--clean", action="store_true", help="wipe the output dir first")
    ap.add_argument("--only", default=None,
                    help="comma-separated motion ids: copy only these motions' assets")
    ap.add_argument("--compress", action="store_true",
                    help="re-encode videos to <=480p CRF30 for deploy size (sources untouched)")
    ap.add_argument("--exclude", default=None,
                    help="comma-separated motion ids whose assets must NOT be copied "
                         "into the site; defaults to build_review_page.EXCLUDE_MOTIONS")
    args = ap.parse_args()

    # Keeping an excluded motion off the page is not enough: its files would still
    # sit at a guessable URL on the deployed site. Drop the assets too.
    if args.exclude is not None:
        excluded = {m.strip() for m in args.exclude.split(",") if m.strip()}
    else:
        try:
            from build_review_page import EXCLUDE_MOTIONS
        except ImportError:
            import sys as _sys
            _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from build_review_page import EXCLUDE_MOTIONS
        excluded = set(EXCLUDE_MOTIONS)

    review_html = os.path.join(args.review_dir, "review.html")
    if not os.path.isfile(review_html):
        raise SystemExit(
            f"{review_html} not found.\n"
            f"Run scripts/render_g1_motion.py then scripts/build_review_page.py first."
        )

    if args.clean and os.path.isdir(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)

    # --- page ------------------------------------------------------------
    shutil.copy2(review_html, os.path.join(args.out, "index.html"))
    copied = [("index.html", os.path.getsize(review_html))]

    # --- intro trailer -----------------------------------------------------
    # The 4K Ultra HD army trailer is the page's introductory hero video. It is
    # copied verbatim, never through the <=480p --compress path: downscaling the
    # one asset whose entire job is image quality would defeat it.
    for f in ("intro_army_trailer.mp4", "intro_army_trailer.jpg"):
        src = os.path.join(args.review_dir, f)
        if os.path.isfile(src):
            dst = os.path.join(args.out, f)
            if not (os.path.exists(dst)
                    and os.path.getmtime(dst) >= os.path.getmtime(src)):
                shutil.copy2(src, dst)
            copied.append((f, os.path.getsize(dst)))
        else:
            print(f"[warn] {src} not found -- the page hides the intro hero "
                  f"until the trailer is rendered and encoded there.")

    only = ({m.strip() for m in args.only.split(",") if m.strip()}
            if args.only else None)

    PREFIXES = ("robot_", "mjc_raw_", "mjc_", "overlay_", "rl_", "diag_",
                "img_raw_", "img_corr_")

    def _motion_of(fname: str) -> str | None:
        stem = os.path.splitext(fname)[0]
        for pfx in PREFIXES:
            if stem.startswith(pfx):
                return stem.removeprefix(pfx)
        return None

    def _wanted(fname: str) -> bool:
        mid = _motion_of(fname)
        if mid is not None and mid in excluded:
            return False
        if only is None:
            return True
        return True if mid is None else mid in only

    # --- robot videos ----------------------------------------------------
    for f in sorted(os.listdir(args.review_dir)):
        if not _wanted(f):
            continue
        if (f.startswith(("robot_", "mjc_", "overlay_", "rl_")) and f.endswith(".mp4")) or \
           (f.startswith("diag_") and f.endswith(".png")):
            src = os.path.join(args.review_dir, f)
            dst = os.path.join(args.out, f)
            if f.endswith(".mp4"):
                copy_video(src, dst, args.compress)
            else:
                shutil.copy2(src, dst)
            copied.append((f, os.path.getsize(dst)))

    # --- human videos ----------------------------------------------------
    # build_review_page.py stages these into <review-dir>/human/ and the page
    # references them as "human/<file>". Copy that exact layout so the deployed
    # URLs resolve; never re-derive paths from --human-dir, or the page and the
    # files end up disagreeing.
    staged = os.path.join(args.review_dir, "human")
    if os.path.isdir(staged):
        dest = os.path.join(args.out, "human")
        os.makedirs(dest, exist_ok=True)
        for f in sorted(os.listdir(staged)):
            stem = os.path.splitext(f)[0]
            if stem in excluded:
                continue
            if only is not None and stem not in only:
                continue
            if f.lower().endswith((".mp4", ".mov", ".webm", ".m4v")):
                src = os.path.join(staged, f)
                dst = os.path.join(dest, f)
                copy_video(src, dst, args.compress)
                copied.append((f"human/{f}", os.path.getsize(dst)))
    else:
        print(f"[warn] no staged human videos at {staged} "
              f"-- the page will show 'No source video paired'.\n"
              f"       Re-run build_review_page.py with --human-dir to stage them.")

    # --- purge excluded motions ------------------------------------------
    # This script only ever copies, so an id added to the exclude list AFTER a
    # previous run would still be sitting in the output directory and would still
    # deploy. Delete those leftovers explicitly.
    purged = []
    for f in sorted(os.listdir(args.out)):
        mid = _motion_of(f)
        if mid is not None and mid in excluded:
            os.remove(os.path.join(args.out, f))
            purged.append(f)
    hdir = os.path.join(args.out, "human")
    if os.path.isdir(hdir):
        for f in sorted(os.listdir(hdir)):
            if os.path.splitext(f)[0] in excluded:
                os.remove(os.path.join(hdir, f))
                purged.append(f"human/{f}")
    if purged:
        print(f"purged {len(purged)} file(s) for excluded motions: {purged}")
    copied = [c for c in copied if c[0] not in set(purged)]

    # --- vercel config ---------------------------------------------------
    vercel_cfg = {
        "version": 2,
        "cleanUrls": True,
        "headers": [
            {
                "source": "/(.*).mp4",
                "headers": [
                    {"key": "Cache-Control", "value": "public, max-age=31536000, immutable"},
                    {"key": "Accept-Ranges", "value": "bytes"},
                ],
            }
        ],
    }
    with open(os.path.join(args.out, "vercel.json"), "w") as fh:
        json.dump(vercel_cfg, fh, indent=2)

    # --- report ----------------------------------------------------------
    total = sum(sz for _n, sz in copied)
    oversized = [(n, sz) for n, sz in copied if sz / 1e6 > MAX_FILE_MB]

    videos = [c for c in copied if c[0].endswith(".mp4")]
    print(f"site written to {args.out}/")
    print(f"  files : {len(copied)}  ({len(videos)} videos)")
    print(f"  total : {human_size(total)}")
    if oversized:
        print(f"\n  OVERSIZED (>{MAX_FILE_MB:.0f} MB, Vercel will reject these):")
        for n, sz in oversized:
            print(f"    {n}  {human_size(sz)}")
        print("  -> re-encode them smaller, e.g.")
        print('     ffmpeg -i in.mp4 -vf "scale=-2:480" -c:v libx264 -crf 30 -an out.mp4')
    if total / 1e6 > WARN_TOTAL_MB:
        print(f"\n  NOTE: {human_size(total)} total. Large video sets can exceed Vercel's")
        print("  deployment limits. If the deploy fails, shrink the videos first:")
        print('     for f in web/*.mp4; do ffmpeg -y -i "$f" -vf "scale=-2:480" '
              '-c:v libx264 -crf 30 -an "$f.tmp.mp4" && mv "$f.tmp.mp4" "$f"; done')

    print(f"\ndeploy with:\n  npx vercel deploy --prod {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
