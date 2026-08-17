#!/usr/bin/env python3
"""Build the side-by-side human-vs-G1 retarget review page.

Produces ONE self-contained HTML file: original human video on the left, the
generated G1 motion on the right, click-through navigation over all motions, and
per-motion verdict controls (Good / Needs correction / Bad) plus free-text notes.
Verdicts are saved to the reviewer's browser (localStorage) and can be exported
as JSON or CSV, so the feedback comes back as data rather than as a conversation.

Pairing rule: a robot video results_review/robot_<id>.mp4 is paired with a human
video whose filename stem is <id> (any common video extension) in --human-dir.
Unpaired entries are still listed and clearly marked, because a missing source
video is itself something the reviewer needs to see.

Usage
    python3 scripts/build_review_page.py --manifest results_review/render_manifest.json \
        --human-dir data/kalari_videos --out results_review/review.html
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import shutil

VIDEO_EXT = (".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv")

# Motions kept out of the published review. The source NPZ/CSV/renders are left
# alone; this only controls what the page shows and what gets deployed.
#   sc_low_squat_drill - the source clip opens on a full-screen troupe logo card
#                        and carries a burned-in caption, so it publishes the
#                        very branding the page is supposed to be free of.
EXCLUDE_MOTIONS = {"sc_low_squat_drill"}

# Kalari technique families, as spelled in the manifest, mapped to what a reviewer
# should actually read in the menu. Keys must match render_manifest.json "family".
FAMILY_LABEL = {
    "vadivu": "Vadivu (stances)",
    "chuvadu": "Chuvadu (footwork)",
    "kicks": "Kicks & leg raises",
    "meypayattu": "Meypayattu (body sequences)",
    "empty_hand": "Empty hand",
    "stable_stance": "Stable stance",
    "translational": "Translational",
    "unlabeled": "Unlabelled",
}
# Order families run top-to-bottom in the menu: stances and footwork are the
# foundation, so they read first; unlabelled sits last.
FAMILY_ORDER = ["vadivu", "chuvadu", "meypayattu", "kicks", "empty_hand",
                "stable_stance", "translational", "unlabeled"]


# Source label per YouTube id. The group comments in kalari_sources.csv name the
# troupe/upload, but only some of them repeat the id in parentheses, so the rest
# would fall back to showing a raw watch id in the menu. These are read off those
# same comments plus the motion_id prefix each batch uses.
VIDEO_SOURCE = {
    "EEbNGDnTiQw": "Kalari Warriors",
    "0XPPpfc4iB4": "Kerala Tourism",
    "yVr3ap7i6W4": "Kerala Tourism",
    "ukY14A1r9gI": "Kalari Sadhana, Mysuru",
    "MCiw878aSjQ": "Nilam Kalari (northern style)",
    "KNwybi0mSV8": "Footwork progression",
    "3Ev_8o8PE0s": "IndiaVideo meypayattu",
    "uHXrxert0Uw": "Vallabhatta",
    "RGFz9VXqlJw": "Ottachuvadu",
    "Z5pYvBwKKOU": "Nilam Kalari, Pakarchakaal",
    "ZC5bWvmc_t4": "Kalari Sadhana, spine drills",
    # No group comment in the CSV names this upload, so it stays explicitly
    # unattributed rather than getting a troupe invented for it.
    "e6X4ZOZGPYw": "Unattributed (e6X4ZOZGPYw)",
    "YNg7QcSazck": "Kerala Tourism, Kaaluyarthi",
    "vewmWFl3rVY": "Kerala Tourism, Kaikuthi",
}

# Words in the notes that describe how the clip was FILMED, not what the body does.
# They belong on the menu's second line, not in the move name.
_FRAMING = ("wide shot", "single performer", "second performer", "solo segment")


def _titlecase(s: str) -> str:
    """'forward bend into kick left' -> 'Forward bend into kick left'."""
    s = s.strip()
    return s[:1].upper() + s[1:] if s else s


def split_note(note: str) -> tuple[str, str]:
    """Split a curated note into (move name, framing qualifier).

    'front high kick - right performer' -> ('Front high kick (right)', '')
    'low stance transition - wide shot' -> ('Low stance transition', 'wide shot')

    Left/right stays welded to the name because two menu rows differ only by it;
    framing notes move to the subtitle because they say nothing about the posture.
    """
    parts = [p.strip() for p in note.split(" - ") if p.strip()]
    base = parts[0] if parts else note.strip()
    keep: list[str] = []
    side = ""

    def strip_framing(text: str) -> str:
        """Peel every trailing framing phrase, dash-separated or not."""
        again = True
        while again:
            again = False
            for ph in _FRAMING:
                m = re.search(r"[\s,]+" + re.escape(ph) + r"$", text, re.I)
                if m:
                    keep.append(ph)
                    text = text[:m.start()].strip()
                    again = True
        return text

    for p in parts[1:]:
        m = re.fullmatch(r"(left|right) performer", p, re.I)
        if m:
            side = m.group(1).lower()
        else:
            keep.append(p.lower() if p.lower() in _FRAMING else p)

    base = strip_framing(base)
    # A trailing 'left'/'right', with or without 'performer', marks which of two
    # performers the clip was cropped to. It has to stay in the name: otherwise
    # the two crops of one move become two identical menu rows.
    m = re.search(r"[\s,]+(left|right)(\s+performer)?$", base, re.I)
    if m:
        side = m.group(1).lower()
        base = base[:m.start()].strip()
    base = strip_framing(base)
    if side:
        base = f"{base} ({side})"

    seen: set[str] = set()
    ordered = [k for k in keep if not (k.lower() in seen or seen.add(k.lower()))]
    return _titlecase(base), ", ".join(ordered)


def load_sources(path: str) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Read data/kalari_sources.csv -> (name, source label, framing qualifier) per id.

    The file is a CSV with '#' comment lines, and each group of clips is preceded
    by a comment naming the troupe/upload it was cut from, e.g.
        # ---- Kalari Warriors instructional (EEbNGDnTiQw): fixed camera, ...
    The 11-character YouTube id in parentheses ties that label to the rows that
    follow, which is the only place the human-readable source name exists.
    """
    names: dict[str, str] = {}
    src: dict[str, str] = {}
    quals: dict[str, str] = {}
    if not os.path.isfile(path):
        return names, src, quals
    vid_label: dict[str, str] = {}
    rows: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.lstrip().startswith("#"):
                m = re.search(r"\(([A-Za-z0-9_-]{11})\)", line)
                if m:
                    lab = line.lstrip("# -=").split("(")[0].strip(" -=")
                    if lab:
                        vid_label[m.group(1)] = lab
                continue
            rows.append(line)
    if not rows:
        return names, src, quals
    for r in csv.DictReader(rows):
        mid = (r.get("motion_id") or "").strip()
        if not mid:
            continue
        note = (r.get("notes") or "").strip()
        if note:
            nm, q = split_note(note)
            names[mid] = nm
            if q:
                quals[mid] = q
        vid = (r.get("video_id") or "").strip()
        if vid:
            # curated label first, then whatever the group comment declared,
            # and only then the bare watch id
            src[mid] = VIDEO_SOURCE.get(vid) or vid_label.get(vid) or vid
    return names, src, quals


def fallback_name(motion_id: str) -> str:
    """Readable name for a motion the sources CSV does not cover.

    Strips the two-letter source prefix ('kw_', 'fw_', ...) that is an internal
    grouping code, not something a reviewer needs to read.
    """
    stem = re.sub(r"^[a-z]{2}_", "", motion_id)
    stem = stem.replace("_", " ")
    stem = re.sub(r"\b([lr])\b", lambda m: "left" if m.group(1) == "l" else "right", stem)
    return _titlecase(stem)


def find_human_videos(human_dir: str | None) -> dict[str, str]:
    if not human_dir or not os.path.isdir(human_dir):
        return {}
    out = {}
    for root, _dirs, files in os.walk(human_dir):
        for f in files:
            stem, ext = os.path.splitext(f)
            if ext.lower() in VIDEO_EXT:
                out.setdefault(stem, os.path.join(root, f))
    return out


def rel(path: str, start: str) -> str:
    return os.path.relpath(path, start).replace(os.sep, "/")


def stage_human_videos(human_map: dict[str, str], out_dir: str) -> dict[str, str]:
    """Copy human videos into <out_dir>/human/ and return stem -> relative URL.

    The page must reference them by a path *inside* the output directory. Linking
    straight at data/kalari_videos/ would produce '../data/...' URLs, which work
    from the local filesystem but can never be deployed, since they point outside
    the web root. Staging a copy makes the review directory self-contained and
    deployable unchanged.
    """
    if not human_map:
        return {}
    dest_dir = os.path.join(out_dir, "human")
    os.makedirs(dest_dir, exist_ok=True)
    out = {}
    for stem, src in human_map.items():
        name = os.path.basename(src)
        dst = os.path.join(dest_dir, name)
        if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
            shutil.copy2(src, dst)
        out[stem] = f"human/{name}"
    return out


def _physics_fields(d: dict | None, out_dir: str) -> dict:
    """Physics-check payload for one motion (from motion_diagnostics.py)."""
    if not d:
        return {"mjc": None, "diag_plot": None, "physics_flags": None, "metrics": None}
    before = d.get("before") or {}
    return {
        "mjc": rel(d["mjc_video"], out_dir) if d.get("mjc_video") else None,
        "diag_plot": rel(d["diag_plot"], out_dir) if d.get("diag_plot") else None,
        "physics_flags": d.get("physics_flags", []),
        "series": d.get("series"),
        # Pre-correction traces: the left-hand ("AI generated") pane must plot the
        # motion it is actually showing, not the corrected one.
        "series_before": before.get("series"),
        "metrics": {
            "penetration_cm": d.get("foot_penetration_max_cm"),
            "float_cm": d.get("foot_float_max_cm"),
            "limit_viol_frames": d.get("joint_limit_violation_frames"),
            "peak_vel": d.get("peak_joint_velocity_rad_s"),
            "peak_vel_joint": d.get("peak_velocity_joint"),
            "before_float_cm": (d.get("before") or {}).get("foot_float_max_cm"),
            "before_pen_cm": (d.get("before") or {}).get("foot_penetration_max_cm"),
            "before_vel": (d.get("before") or {}).get("peak_joint_velocity_rad_s"),
        },
    }


def _cmp_fields(d: dict | None, out_dir: str) -> dict:
    """Fixed-camera raw/corrected/overlay renders (scripts/render_compare.py)."""
    if not d:
        return {"raw_mjc": None, "overlay": None}
    return {
        "raw_mjc": rel(d["raw_video"], out_dir) if d.get("raw_video") else None,
        "overlay": rel(d["overlay_video"], out_dir) if d.get("overlay_video") else None,
    }


def _rl_fields(d: dict | None, out_dir: str) -> dict:
    """Stage A RL payload for one motion (from scripts/ingest_rl_results.py)."""
    if not d:
        return {"rl": None, "rl_status": None, "rl_metrics": None}
    return {
        "rl": rel(d["video"], out_dir) if d.get("video") else None,
        "rl_status": d.get("status_note"),
        "rl_metrics": d.get("metrics"),
    }


def build(records: list[dict], human_map: dict[str, str], out_path: str,
          title: str, require_human: bool = False,
          diagnostics: dict | None = None, rl: dict | None = None,
          cmp_map: dict | None = None, names: dict | None = None,
          srcs: dict | None = None, quals: dict | None = None) -> tuple[str, int]:
    diagnostics = diagnostics or {}
    rl = rl or {}
    cmp_map = cmp_map or {}
    names = names or {}
    srcs = srcs or {}
    quals = quals or {}
    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    human_urls = stage_human_videos(human_map, out_dir)
    items = []
    paired = 0
    matched_stems: set[str] = set()
    dropped: list[str] = []
    for r in records:
        mid = r["motion_id"]
        human_stem = mid if mid in human_map else None
        if human_stem is None:
            norm = mid.lower().replace("-", "_")
            for key in human_map:
                if key.lower().replace("-", "_") == norm:
                    human_stem = key
                    break
        if human_stem:
            paired += 1
            matched_stems.add(human_stem)
        elif require_human:
            # A robot motion with no corresponding human video cannot be judged
            # against anything, and sitting in a retarget review it reads as a
            # conversion of footage that was never supplied. Drop it.
            dropped.append(mid)
            continue
        items.append({
            "id": mid,
            "name": names.get(mid) or fallback_name(mid),
            "qualifier": quals.get(mid, ""),
            "family": r.get("family", "unlabeled"),
            "source": r.get("source", ""),
            "robot": rel(r["robot_video"], out_dir) if r.get("robot_video") else None,
            "human": human_urls.get(human_stem) if human_stem else None,
            "frames": r.get("frames"),
            "fps": r.get("fps"),
            "duration": r.get("duration_s"),
            "joints_matched": r.get("joints_matched"),
            "joints_expected": r.get("joints_expected"),
            "flags": r.get("quality_flags", []),
            "render_label": r.get("render_label", ""),
            "ground_clamped": r.get("ground_clamped", False),
            **_physics_fields(diagnostics.get(mid), out_dir),
            **_cmp_fields(cmp_map.get(mid), out_dir),
        })

    # Human videos with no robot render at all: the retarget either was never run
    # or failed. These MUST still appear -- a silently missing motion looks like a
    # motion that was never requested, which is exactly what a review must not hide.
    for stem, _path in sorted(human_map.items()):
        if stem in matched_stems:
            continue
        items.append({
            "id": stem,
            "name": names.get(stem) or fallback_name(stem),
            "qualifier": quals.get(stem, ""),
            "family": "unlabeled",
            "source": "human video only",
            "robot": None,
            "human": human_urls[stem],
            "frames": None, "fps": None, "duration": None,
            "joints_matched": None, "joints_expected": None,
            "flags": ["no robot render - retarget not run or failed"],
            "render_label": "",
            "ground_clamped": False,
            **_physics_fields(diagnostics.get(stem), out_dir),
            **_cmp_fields(cmp_map.get(stem), out_dir),
        })
        paired += 0  # counted as present, but not a human/robot pair

    if dropped:
        print(f"dropped {len(dropped)} robot motion(s) with no matching human video: {dropped}")

    payload = json.dumps(items, indent=None)
    fam_cfg = json.dumps({"label": FAMILY_LABEL, "order": FAMILY_ORDER})
    doc = _TEMPLATE.replace("__TITLE__", html.escape(title)) \
                   .replace("__FAMILIES__", fam_cfg) \
                   .replace("__DATA__", payload) \
                   .replace("__TOTAL__", str(len(items))) \
                   .replace("__PAIRED__", str(paired))
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(doc)
    return out_path, paired


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>__TITLE__</title>
<style>
  :root{
    --bg:#f4f1e8; --paper:#fffdf7; --ink:#1c2322; --muted:#65706d;
    --line:#ded5c4; --accent:#8d2f23; --good:#1d6b60; --warn:#b8860b; --bad:#a12d20;
    --shadow:0 10px 30px rgba(48,31,18,.08);
  }
  :root{ --chartline:#2563b8; --chartline2:#1d6b60; }
  /* The four review markings. Used identically in the plots, the overlay legend
     and the contact ribbons, so a colour means one thing everywhere on the page. */
  :root{ --mk-raw:#c0392b; --mk-corr:#1d8a5f; --mk-float:#d99413; --mk-pen:#2f6fd0; }
  @media (prefers-color-scheme: dark){
    :root{ --bg:#14181a; --paper:#1c2124; --ink:#e8e4dc; --muted:#98a2a0;
           --line:#2e3538; --accent:#e0705f; --good:#4fbfae; --warn:#d9a441; --bad:#e0705f;
           --chartline:#4d8ce0; --chartline2:#4fbfae;
           --mk-raw:#e0705f; --mk-corr:#4fbfae; --mk-float:#e8b444; --mk-pen:#6da4ee;
           --shadow:0 10px 30px rgba(0,0,0,.35); }
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:15px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Arial,sans-serif}
  header{position:sticky;top:0;z-index:10;background:var(--paper);
         border-bottom:1px solid var(--line);padding:14px 20px;box-shadow:var(--shadow)}
  h1{margin:0 0 4px;font-size:19px;letter-spacing:.2px}
  .sub{color:var(--muted);font-size:13px}
  .wrap{display:grid;grid-template-columns:290px minmax(0,1fr);gap:0;min-height:calc(100vh - 74px)}
  nav{border-right:1px solid var(--line);background:var(--paper);overflow-y:auto;
      max-height:calc(100vh - 74px);position:sticky;top:74px}
  .navitem{padding:7px 12px 7px 14px;border-bottom:1px solid var(--line);cursor:pointer;font-size:13px;
           display:flex;gap:8px;align-items:flex-start}
  .navitem:hover{background:var(--bg)}
  .navitem.active{background:var(--bg);border-left:3px solid var(--accent);padding-left:11px}
  .navitem.active .nm{font-weight:650}
  .navitem .nm{flex:1;min-width:0;overflow:hidden}
  .navitem .t{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .navitem .sub2{font-size:10.5px;color:var(--muted);overflow:hidden;
                 text-overflow:ellipsis;white-space:nowrap;margin-top:1px}
  .navitem .idx{color:var(--muted);font-size:11px;flex:0 0 auto;padding-top:1px;
                font-variant-numeric:tabular-nums}
  /* Deliberately not sticky: the filter box above already sticks at top:0, and a
     second sticky at the same offset parks the group header behind it. */
  .navgrp{padding:8px 14px 5px;font-size:10.5px;font-weight:700;letter-spacing:.7px;
          text-transform:uppercase;color:var(--accent);background:var(--bg);
          border-bottom:1px solid var(--line)}
  .navgrp span{color:var(--muted);font-weight:500;letter-spacing:.2px}
  #navFilter{width:100%;padding:7px 10px;border:1px solid var(--line);border-radius:7px;
             background:var(--bg);color:var(--ink);font:12.5px/1.3 inherit}
  .navfind{padding:10px 12px;border-bottom:1px solid var(--line);position:sticky;top:0;
           background:var(--paper);z-index:3}
  main{padding:20px 24px;min-width:0}
  /* Three stacked rows: (1) source video, (2) AI vs corrected side by side with a
     plot under each, (3) the overlay. Only row 2 is two-column. */
  .stage{display:flex;flex-direction:column;gap:16px}
  /* minmax(0,1fr), not 1fr: the plot <svg> carries a 600-unit viewBox, and with a
     plain 1fr its min-content contribution can force each column wider than the
     available space, pushing the corrected pane out of view so Row 2 stops
     reading as two side-by-side bars. minmax(0,...) lets the columns shrink. */
  .row.duo{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px;align-items:start}
  .row.duo>.pane{min-width:0}
  .center-media{max-width:620px;margin:0 auto}
  .center-media.wide{max-width:760px}
  /* Collapse to one column only on genuinely narrow screens. At 900px a normal
     half-width desktop window still stacked Row 2, which looks like the layout
     was never applied. */
  @media(max-width:1000px){ .wrap{grid-template-columns:minmax(0,1fr)} nav{position:static;max-height:260px} }
  @media(max-width:680px){ .row.duo{grid-template-columns:minmax(0,1fr)} }
  /* Row 2B's heading is longer than 2A's and wraps to two lines. Without a floor
     on both, the two videos start at different heights and stop being comparable
     at a glance, which is the whole point of putting them side by side. */
  .row.duo .pane h3{min-height:2.9em}
  .rn{display:inline-block;background:var(--accent);color:#fff;border-radius:4px;
      padding:1px 6px;font-size:10px;font-weight:700;letter-spacing:.4px;margin-right:7px}
  .sw{display:inline-block;width:10px;height:10px;border-radius:2px;vertical-align:-1px;margin-right:6px}
  .sw-raw{background:var(--mk-raw)} .sw-corr{background:var(--mk-corr)}
  .sw-float{background:var(--mk-float)} .sw-pen{background:var(--mk-pen)}
  .plots{margin-top:10px;border-top:1px solid var(--line);padding-top:9px}
  .plots .ptitle{font-size:10.5px;color:var(--muted);margin:0 0 1px}
  .legend{display:flex;flex-wrap:wrap;gap:12px;margin:10px 0 2px;font-size:11.5px;color:var(--muted)}
  .legend b{color:var(--ink);font-weight:600}
  .ribbon{margin-top:9px}
  .ribbon .rlab{font-size:10.5px;font-weight:700;letter-spacing:.4px;margin-bottom:2px}
  .pane{background:var(--paper);border:1px solid var(--line);border-radius:10px;
        padding:12px;box-shadow:var(--shadow)}
  .pane h3{margin:0 0 8px;font-size:13px;text-transform:uppercase;letter-spacing:.6px;color:var(--muted)}
  video{width:100%;border-radius:7px;background:#000;display:block}
  .missing{aspect-ratio:4/3;display:grid;place-items:center;border:1px dashed var(--line);
           border-radius:7px;color:var(--muted);font-size:13px;text-align:center;padding:16px}
  .meta{margin-top:14px;background:var(--paper);border:1px solid var(--line);
        border-radius:10px;padding:14px;box-shadow:var(--shadow)}
  .chips{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 0}
  .chip{font-size:11.5px;padding:3px 9px;border-radius:999px;border:1px solid var(--line);color:var(--muted)}
  .chip.flag{border-color:var(--warn);color:var(--warn)}
  .verdict{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 8px}
  button.v{border:1px solid var(--line);background:transparent;color:var(--ink);
           padding:8px 15px;border-radius:8px;cursor:pointer;font-size:13px;font-weight:600}
  button.v:hover{border-color:var(--accent)}
  button.v.sel[data-v="good"]{background:var(--good);border-color:var(--good);color:#fff}
  button.v.sel[data-v="fix"]{background:var(--warn);border-color:var(--warn);color:#fff}
  button.v.sel[data-v="bad"]{background:var(--bad);border-color:var(--bad);color:#fff}
  textarea{width:100%;min-height:64px;padding:9px;border:1px solid var(--line);border-radius:8px;
           background:var(--bg);color:var(--ink);font:13px/1.45 inherit;resize:vertical}
  .dot{width:9px;height:9px;border-radius:50%;flex:0 0 9px;background:var(--line)}
  .dot.good{background:var(--good)} .dot.fix{background:var(--warn)} .dot.bad{background:var(--bad)}
  .bar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:8px}
  .bar button{border:1px solid var(--line);background:transparent;color:var(--ink);
              padding:6px 12px;border-radius:7px;cursor:pointer;font-size:12.5px}
  .note{font-size:12px;color:var(--muted);margin-top:10px;line-height:1.5}
  kbd{border:1px solid var(--line);border-radius:4px;padding:1px 5px;font-size:11px;background:var(--bg)}
  button.ovm{border:1px solid var(--line);background:transparent;color:var(--ink);
             padding:3px 10px;border-radius:6px;font-size:11.5px;cursor:pointer}
  button.ovm.sel{background:var(--accent);border-color:var(--accent);color:#fff}
  #ovBox .wipedivider{position:absolute;top:0;bottom:0;width:3px;background:#fff;
    box-shadow:0 0 4px rgba(0,0,0,.6);cursor:ew-resize;z-index:5}
  #ovBox .wipedivider::after{content:"◀ ▶";position:absolute;top:50%;left:50%;
    transform:translate(-50%,-50%);background:#fff;color:#222;font-size:9px;
    padding:2px 4px;border-radius:8px;white-space:nowrap}
  #ovBox .ovtag{position:absolute;top:6px;font-size:10px;font-weight:700;color:#fff;
    background:rgba(0,0,0,.55);padding:2px 7px;border-radius:4px;z-index:4}

  /* ================= Intro trailer hero ================= */
  #hero{background:linear-gradient(180deg,#0a0d12,#121821);border-bottom:1px solid var(--line)}
  #introWrap{position:relative;max-width:1720px;margin:0 auto;background:#000}
  #introVid{width:100%;max-height:78vh;object-fit:cover;display:block;background:#000}
  #introOverlay{position:absolute;left:0;right:0;bottom:0;padding:22px 28px 18px;
    background:linear-gradient(0deg,rgba(4,7,11,.78),rgba(4,7,11,0));display:flex;
    align-items:flex-end;justify-content:space-between;gap:14px;flex-wrap:wrap;pointer-events:none}
  #introOverlay h2{margin:0;font-size:clamp(20px,3.2vw,34px);letter-spacing:5px;
    font-weight:800;color:#f4f7fb}
  #introOverlay .tag{font-size:12px;letter-spacing:1.8px;color:#c3cedd;text-transform:uppercase}
  .uhdchip{display:inline-block;border:1.5px solid rgba(255,255,255,.55);border-radius:5px;
    padding:1px 8px;font-size:11px;font-weight:800;letter-spacing:1.2px;color:#fff;
    vertical-align:6px;margin-left:12px}
  #introBtns{display:flex;gap:8px;pointer-events:auto}
  #introBtns button{border:1px solid rgba(255,255,255,.4);background:rgba(8,12,18,.5);
    color:#fff;padding:7px 15px;border-radius:999px;font-size:12px;cursor:pointer}
  #introBtns button:hover{background:rgba(255,255,255,.2)}

  /* ================= Interactive training floor ================= */
  #floorSec{background:var(--paper);border-bottom:1px solid var(--line);
    padding:24px 20px 30px;overflow:hidden}
  .floorhead{max-width:1100px;margin:0 auto 6px;display:flex;align-items:baseline;
    gap:14px;flex-wrap:wrap}
  .floorhead h2{margin:0;font-size:16px;letter-spacing:.6px;text-transform:uppercase}
  .floorhead .hint{color:var(--muted);font-size:12.5px;flex:1}
  .floorhead button{border:1px solid var(--line);background:transparent;color:var(--ink);
    padding:5px 12px;border-radius:7px;font-size:12px;cursor:pointer}
  .floorhead button:hover{border-color:var(--accent)}
  #floorStage{max-width:1100px;height:430px;margin:0 auto;perspective:1500px;
    perspective-origin:50% 30%;cursor:grab;touch-action:none;user-select:none;
    -webkit-user-select:none}
  #floorStage.grabbing{cursor:grabbing}
  #floorPlane{--rx:57deg;--rz:-24deg;position:relative;width:880px;height:340px;
    margin:70px auto 0;transform-style:preserve-3d;
    transform:rotateX(var(--rx)) rotateZ(var(--rz));transition:transform .08s linear}
  #floorPlane .mat{position:absolute;inset:-60px -40px;border-radius:18px;
    background:linear-gradient(135deg,#fbfbfd 0%,#eef0f5 55%,#f7f8fb 100%);
    box-shadow:0 40px 80px rgba(20,26,40,.35), inset 0 0 60px rgba(255,255,255,.85);
    border:1px solid #dfe3ec}
  @media (prefers-color-scheme: dark){
    #floorPlane .mat{background:linear-gradient(135deg,#262b33 0%,#1b2027 55%,#232830 100%);
      border-color:#343b45;box-shadow:0 40px 80px rgba(0,0,0,.6), inset 0 0 60px rgba(255,255,255,.05)}
  }
  .zone3d{position:absolute;border:3px solid #2a3347;border-radius:4px;
    transform-style:preserve-3d;cursor:pointer;transition:box-shadow .18s,filter .18s}
  .zone3d .zlabel{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);
    font-size:20px;font-weight:800;letter-spacing:5px;color:#2a3347;opacity:.65;
    white-space:nowrap;pointer-events:none}
  .zone3d.hot{box-shadow:0 0 0 3px var(--accent), 0 0 34px rgba(141,47,35,.35);
    filter:brightness(1.04)}
  .zone3d.picked{box-shadow:0 0 0 3px var(--accent), 0 0 26px rgba(141,47,35,.5)}
  .zone3d .tick{position:absolute;background:#2a3347;pointer-events:none}
  .bot{position:absolute;width:0;height:0;transform-style:preserve-3d;pointer-events:none}
  .bot .shadow{position:absolute;left:-7px;top:-4px;width:14px;height:8px;
    border-radius:50%;background:radial-gradient(ellipse,rgba(25,30,45,.38),transparent 70%)}
  .bot .fig{position:absolute;left:-8px;top:-26px;width:16px;height:26px;
    transform-origin:50% 100%;
    transform:rotateZ(calc(-1 * var(--rz))) rotateX(calc(-1 * var(--rx)))}
  .bot .fig svg{width:100%;height:100%;display:block;
    filter:drop-shadow(0 1px 1px rgba(20,26,40,.35))}
  #zoneTip{position:fixed;z-index:20;pointer-events:none;background:var(--paper);
    border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:8px;
    padding:8px 12px;font-size:12px;box-shadow:var(--shadow);display:none;max-width:230px}
  #zoneTip b{display:block;font-size:13px;letter-spacing:.8px}
  #zoneTip .zn{color:var(--muted);margin-top:2px}
  #zoneCards{max-width:1100px;margin:14px auto 0;display:grid;
    grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}
  @media(max-width:680px){ #zoneCards{grid-template-columns:minmax(0,1fr)} #floorStage{height:340px} }
  .zcard{border:1px solid var(--line);border-radius:10px;padding:10px 13px;cursor:pointer;
    background:var(--bg);transition:border-color .15s}
  .zcard:hover,.zcard.picked{border-color:var(--accent)}
  .zcard b{font-size:12.5px;letter-spacing:1px}
  .zcard .zn{font-size:11.5px;color:var(--muted);margin-top:2px}
  .zcard .cnt{float:right;font-size:11px;color:var(--accent);font-weight:700}
  #zoneClear{display:none;margin-top:7px;width:100%;border:1px dashed var(--accent);
    background:transparent;color:var(--accent);padding:6px;border-radius:7px;
    font-size:12px;cursor:pointer}
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="sub">
    <strong>__TOTAL__</strong> motions | <strong>__PAIRED__</strong> paired with source video
    | <span id="counts"></span>
  </div>
  <div class="bar">
    <button onclick="exportJSON()">Export JSON</button>
    <button onclick="exportCSV()">Export CSV</button>
    <button onclick="jump(-1)">&larr; Prev</button>
    <button onclick="jump(1)">Next &rarr;</button>
    <button onclick="if(confirm('Clear all verdicts?'))clearAll()">Clear</button>
    <span class="note" style="margin:0">Keys: <kbd>&larr;</kbd> <kbd>&rarr;</kbd> navigate,
      <kbd>1</kbd> good, <kbd>2</kbd> needs fix, <kbd>3</kbd> bad</span>
  </div>
</header>

<!-- ============ Intro trailer: 4K Ultra HD army showcase ============ -->
<section id="hero">
  <div id="introWrap">
    <video id="introVid" autoplay muted loop playsinline preload="auto"
           poster="intro_army_trailer.jpg">
      <source src="intro_army_trailer.mp4" type="video/mp4">
    </video>
    <div id="introOverlay">
      <div>
        <div class="tag">Introducing</div>
        <h2>KALARISENA<span class="uhdchip">4K ULTRA HD</span></h2>
        <div class="tag">30 Unitree G1 humanoids &middot; Kalaripayattu combat training facility</div>
      </div>
      <div id="introBtns">
        <button id="introPause" title="Pause / play">&#10074;&#10074; Pause</button>
        <button id="introFS" title="Watch fullscreen">&#x26F6; Fullscreen</button>
      </div>
    </div>
  </div>
</section>

<!-- ============ Interactive training floor ============ -->
<section id="floorSec">
  <div class="floorhead">
    <h2>Interactive Training Floor</h2>
    <span class="hint">Drag to orbit the arena &middot; hover a training box &middot; click a box to filter the move list below</span>
    <button id="floorReset">Reset view</button>
  </div>
  <div id="floorStage">
    <div id="floorPlane">
      <div class="mat"></div>
    </div>
  </div>
  <div id="zoneTip"></div>
  <div id="zoneCards"></div>
</section>

<div class="wrap">
  <nav>
    <div class="navfind">
      <input id="navFilter" type="search" autocomplete="off"
             placeholder="Filter by move, family or source...">
      <div style="font-size:10.5px;color:var(--muted);margin-top:5px" id="navCount"></div>
      <button id="zoneClear" onclick="setZone('')">&#10005; Clear floor-box filter</button>
    </div>
    <div id="nav"></div>
  </nav>
  <main>
    <div class="stage">

      <section class="row">
        <div class="pane">
          <h3><span class="rn">ROW 1</span>Original Kalaripayattu Session</h3>
          <div class="center-media"><div id="humanBox"></div></div>
        </div>
      </section>

      <section class="row duo">
        <div class="pane">
          <h3><span class="rn">ROW 2A</span><span class="sw sw-raw"></span>AI generated &mdash; Raw Retargeted</h3>
          <div id="robotBox"></div>
          <div class="plots" id="plotsRaw"></div>
        </div>
        <div class="pane">
          <h3><span class="rn">ROW 2B</span><span class="sw sw-corr"></span>Pragya Physics Corrected - Foot Stability &amp; COM Stability</h3>
          <div id="mjcBox"></div>
          <div class="plots" id="plotsCorr"></div>
        </div>
      </section>

      <section class="row">
        <div class="pane">
          <h3><span class="rn">ROW 3</span>Overlay &mdash; Raw vs. Pragya Corrected, same fixed camera</h3>
          <div id="ovModes" style="display:flex;gap:6px;margin-bottom:8px">
            <button class="ovm sel" data-m="ghost">Ghost</button>
            <button class="ovm" data-m="wipe">Wipe</button>
            <button class="ovm" data-m="fade">Fade</button>
          </div>
          <div class="center-media wide">
            <div id="ovBox" style="position:relative;overflow:hidden;border-radius:7px"></div>
            <input type="range" id="ovFade" min="0" max="100" value="50"
                   style="width:100%;margin-top:6px;display:none">
          </div>
          <div class="legend">
            <span><span class="sw sw-raw"></span><b>Red</b>: AI generated (raw), the ghost figure</span>
            <span><span class="sw sw-corr"></span><b>Green</b>: Pragya corrected, the solid figure &amp; frames in ground contact</span>
            <span><span class="sw sw-float"></span><b>Yellow</b>: foot floating, contact lost</span>
            <span><span class="sw sw-pen"></span><b>Blue</b>: foot below floor, ground penetration</span>
          </div>
          <div class="plots" id="ovCombo"></div>
          <div id="ovRibbons"></div>
          <div id="ovCap" style="font-size:11px;color:var(--muted);margin-top:7px"></div>
        </div>
      </section>
    </div>
    <div class="meta" id="physBlock" style="display:none">
      <div style="font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:.6px;color:var(--muted);margin-bottom:6px">
        Physics diagnostics: raw retarget &rarr; after ground correction (measured)</div>
      <div class="chips" id="physChips"></div>
      <div class="note" style="font-size:11.5px;color:var(--muted);margin-top:8px">
        The plot under each Row 2 video draws itself as that video plays, so the trace and the
        motion advance together. Click anywhere in a plot to jump every video to that moment.</div>
      <img id="physPlot" style="max-width:100%;border-radius:8px;margin-top:10px;display:none" alt=""/>
    </div>
    <!-- kept outside #physBlock: that block is hidden for motions without metrics,
         and a hidden parent would swallow the plot tooltip with it. -->
    <div id="chartTip" style="position:fixed;display:none;pointer-events:none;background:var(--paper);border:1px solid var(--line);border-radius:6px;padding:4px 8px;font-size:11.5px;box-shadow:var(--shadow);z-index:50"></div>
    <div class="meta">
      <div style="font-size:17px;font-weight:600" id="mTitle">-</div>
      <div class="chips" id="mChips"></div>
      <div class="verdict">
        <button class="v" data-v="good" onclick="setV('good')">Good</button>
        <button class="v" data-v="fix"  onclick="setV('fix')">Needs correction</button>
        <button class="v" data-v="bad"  onclick="setV('bad')">Bad / unusable</button>
      </div>
      <textarea id="notes" placeholder="What is wrong? e.g. arms too low, foot sliding, stance not deep enough..."></textarea>
      <div class="note" id="renderNote"></div>
    </div>
  </main>
</div>
<script>
const DATA = __DATA__;
const KEY = "kalarisena_review_v1";
let idx = 0;
let store = {};
try { store = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) { store = {}; }

function save(){ localStorage.setItem(KEY, JSON.stringify(store)); renderNav(); counts(); }
function rec(id){ return store[id] || (store[id] = {verdict:"", notes:""}); }

function counts(){
  const c={good:0,fix:0,bad:0,none:0};
  DATA.forEach(d=>{ const v=(store[d.id]||{}).verdict; c[v||"none"]=(c[v||"none"]||0)+1; });
  document.getElementById("counts").textContent =
    `reviewed ${DATA.length-c.none}/${DATA.length}: ${c.good} good, ${c.fix} needs fix, ${c.bad} bad`;
}

const FAMILIES = __FAMILIES__;
let navQuery = "";
let navZone = "";      // set by clicking a training box on the interactive floor

// Same routing rule the trailer renderer uses (render_army_trailer.classify), so
// a box on the interactive floor filters to exactly the moves that train there.
function zoneOf(d){
  const n = (d.id + " " + d.name).toLowerCase();
  if(n.includes("block") || (n.includes("guard") && !n.includes("kick"))) return "defense";
  if(/kick|strike|jump|leap/.test(n)) return "attack";
  return "beginner";
}
function famLabel(f){ return (FAMILIES.label && FAMILIES.label[f]) || f; }
function esc(s){ return String(s==null?"":s)
  .replace(/[&<>"]/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }

// Menu shows the curated move name, grouped by Kalari family, with the source
// troupe underneath. The motion_id is an internal handle, so it moves to the
// hover tooltip rather than being the thing a reviewer has to read.
function renderNav(){
  const q = navQuery.trim().toLowerCase();
  const match = d => (!navZone || zoneOf(d) === navZone) &&
        (!q || [d.name, d.id, d.family, famLabel(d.family)]
        .some(x => String(x||"").toLowerCase().includes(q)));
  const order = FAMILIES.order.slice();
  DATA.forEach(d => { if(order.indexOf(d.family) < 0) order.push(d.family); });
  let out = "", shown = 0;
  order.forEach(fam => {
    const rows = DATA.map((d,i)=>({d,i})).filter(x => x.d.family === fam && match(x.d));
    if(!rows.length) return;
    out += `<div class="navgrp">${esc(famLabel(fam))} <span>&middot; ${rows.length}</span></div>`;
    rows.forEach(({d,i}) => {
      shown++;
      const v = (store[d.id]||{}).verdict || "";
      const sub = d.qualifier || "";
      out += `<div class="navitem ${i===idx?"active":""}" onclick="go(${i})" title="${esc(d.id)}">
        <span class="idx">${i+1}</span>
        <span class="dot ${v}" style="margin-top:4px"></span>
        <span class="nm"><div class="t">${esc(d.name)}</div>
          <div class="sub2">${esc(sub)}${d.flags.length?" &middot; ⚑":""}</div>
        </span></div>`;
    });
  });
  document.getElementById("nav").innerHTML = out ||
    `<div style="padding:14px;color:var(--muted);font-size:12.5px">No move matches that filter.</div>`;
  document.getElementById("navCount").textContent =
    navZone ? `${shown} of ${DATA.length} moves · ${navZone.toUpperCase()} floor box` :
    q ? `${shown} of ${DATA.length} moves` : `${DATA.length} moves in ${order.filter(f=>DATA.some(d=>d.family===f)).length} families`;
  const zc = document.getElementById("zoneClear");
  if(zc) zc.style.display = navZone ? "block" : "none";
}
document.getElementById("navFilter").addEventListener("input", e => {
  navQuery = e.target.value; renderNav();
});

function vid(src, missingMsg){
  return src ? `<video src="${src}" controls autoplay loop muted playsinline></video>`
             : `<div class="missing">${missingMsg}</div>`;
}

function show(){
  const d = DATA[idx];
  document.getElementById("humanBox").innerHTML =
    vid(d.human, "No source video paired.<br>Place it in the human video folder named <b>"+d.id+".mp4</b>");
  document.getElementById("robotBox").innerHTML =
    vid(d.raw_mjc || d.robot, "Conversion pending - queued for GPU batch.");
  document.getElementById("mjcBox").innerHTML =
    vid(d.mjc, "Correction pending - runs after conversion.");
  const ov = document.getElementById("ovBox");
  if (d.overlay && d.raw_mjc && d.mjc) {
    ov.innerHTML = `
      <video id="ovGhost" src="${d.overlay}" controls autoplay loop muted playsinline style="display:block;width:100%"></video>
      <div id="ovStack" style="display:none;position:relative">
        <video id="ovBase" src="${d.mjc}" autoplay loop muted playsinline style="display:block;width:100%"></video>
        <video id="ovTop" src="${d.raw_mjc}" autoplay loop muted playsinline
               style="position:absolute;inset:0;width:100%;pointer-events:none"></video>
        <span class="ovtag" id="tagL" style="left:8px;background:var(--mk-raw)">RAW: AI generated</span>
        <span class="ovtag" id="tagR" style="right:8px;background:var(--mk-corr)">CORRECTED: Pragya</span>
        <div class="wipedivider" id="ovDiv" style="left:50%"></div>
      </div>`;
    setOvMode(ovMode);
  } else {
    ov.innerHTML = `<div class="missing">Overlay pending - needs both raw and corrected renders.</div>`;
    document.getElementById("ovCap").textContent = "";
  }

  const pb = document.getElementById("physBlock");
  if (d.metrics) {
    pb.style.display = "";
    const m = d.metrics;
    const pc = [];
    const chip = (label, warn) => `<span class="chip${warn ? " flag" : ""}">${label}</span>`;
    const ba = (now, before, unit) =>
      before != null && before !== now ? `${before} \u2192 ${now} ${unit}` : `${now} ${unit}`;
    pc.push(chip(`penetration ${ba(m.penetration_cm, m.before_pen_cm, "cm")}`, m.penetration_cm > 2));
    pc.push(chip(`foot float ${ba(m.float_cm, m.before_float_cm, "cm")}`, m.float_cm > 6));
    pc.push(chip(`joint-limit violations ${m.limit_viol_frames} frames`, m.limit_viol_frames > 0));
    pc.push(chip(`peak joint vel ${ba(m.peak_vel, m.before_vel, "rad/s")} (${m.peak_vel_joint})`, m.peak_vel > 12));
    (d.physics_flags || []).forEach(f => pc.push(chip("\u2691 " + f, true)));

    document.getElementById("physChips").innerHTML = pc.join("");
    const img = document.getElementById("physPlot");
    if (d.diag_plot) { img.src = d.diag_plot; img.style.display = ""; }
    else { img.style.display = "none"; }
  } else {
    pb.style.display = "none";
  }
  document.getElementById("mTitle").innerHTML =
    `${idx+1}/${DATA.length} &nbsp;&middot;&nbsp; ${esc(d.name)}` +
    `<div style="font-size:12px;font-weight:400;color:var(--muted);margin-top:3px">` +
    `${esc(famLabel(d.family))}` +
    ` &nbsp;&middot;&nbsp; <code>${esc(d.id)}</code></div>`;

  const chips = [];
  chips.push(`<span class="chip">${esc(famLabel(d.family))}</span>`);
  if(d.qualifier) chips.push(`<span class="chip">${esc(d.qualifier)}</span>`);
  chips.push(`<span class="chip">pipeline: ${esc(d.source)}</span>`);
  if(d.frames) chips.push(`<span class="chip">${d.frames} frames, ${d.duration}s @ ${d.fps}fps</span>`);
  if(d.joints_expected) chips.push(`<span class="chip">joints ${d.joints_matched}/${d.joints_expected}</span>`);
  if(d.ground_clamped) chips.push(`<span class="chip">ground-clamped</span>`);
  d.flags.forEach(f=>chips.push(`<span class="chip flag">⚑ ${f}</span>`));
  document.getElementById("mChips").innerHTML = chips.join("");

  document.getElementById("renderNote").textContent =
    d.render_label ? "Robot pane: " + d.render_label + ". It shows what the retarget produced, not that the robot can physically execute it." : "";

  const r = rec(d.id);
  document.getElementById("notes").value = r.notes || "";
  document.querySelectorAll("button.v").forEach(b=>
    b.classList.toggle("sel", b.dataset.v === r.verdict));
  renderNav();
  counts();
  renderCharts(d);
}

function go(i){ idx = Math.max(0, Math.min(DATA.length-1, i)); show(); }
function jump(d){ go(idx+d); }
function setV(v){ const r=rec(DATA[idx].id); r.verdict = (r.verdict===v?"":v); save(); show(); }
function clearAll(){ store={}; save(); show(); }

document.getElementById("notes").addEventListener("input", e=>{
  rec(DATA[idx].id).notes = e.target.value;
  localStorage.setItem(KEY, JSON.stringify(store));
});

document.addEventListener("keydown", e=>{
  // INPUT matters as much as TEXTAREA now: without it, typing "1" in the menu
  // filter would stamp a "good" verdict and the arrow keys would change motion.
  if(e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT") return;
  if(e.key==="ArrowLeft") jump(-1);
  if(e.key==="ArrowRight") jump(1);
  if(e.key==="1") setV("good");
  if(e.key==="2") setV("fix");
  if(e.key==="3") setV("bad");
});

function rows(){
  return DATA.map(d=>{
    const r = store[d.id] || {};
    return {motion_id:d.id, move:d.name, family:d.family, family_label:famLabel(d.family),
            verdict:r.verdict||"", notes:r.notes||"",
            flags:d.flags.join(" | "), has_human_video:!!d.human, frames:d.frames};
  });
}
function dl(name, text, type){
  const a=document.createElement("a");
  a.href=URL.createObjectURL(new Blob([text],{type}));
  a.download=name; a.click(); URL.revokeObjectURL(a.href);
}
function exportJSON(){ dl("kalari_review.json", JSON.stringify(rows(),null,2), "application/json"); }
function exportCSV(){
  const rs=rows(), cols=Object.keys(rs[0]);
  const esc=v=>`"${String(v).replace(/"/g,'""')}"`;
  dl("kalari_review.csv",
     [cols.join(","), ...rs.map(r=>cols.map(c=>esc(r[c])).join(","))].join("\n"),
     "text/csv");
}

// ---------- overlay modes: ghost / wipe / fade ----------
let ovMode = "ghost";
const OV_CAPTIONS = {
  ghost: "Ghost view: the solid figure is Pragya's corrected motion (green); the RED ghost is the raw AI generation. Where the red ghost floats above the floor (the yellow stretches in the RAW ribbon below), the correction pulled the robot down onto real ground contact.",
  wipe:  "Wipe view: drag the divider. LEFT of the line = raw AI generation (red), RIGHT = corrected (green). Sweep across the feet to watch the ground contact being fixed.",
  fade:  "Fade view: slide between raw (left end) and corrected (right end). Watch the feet and the overall height shift as the correction takes effect.",
};
function setOvMode(m){
  ovMode = m;
  document.querySelectorAll(".ovm").forEach(b => b.classList.toggle("sel", b.dataset.m === m));
  const ghost = document.getElementById("ovGhost"), stack = document.getElementById("ovStack"),
        fade = document.getElementById("ovFade"), div = document.getElementById("ovDiv");
  if (!ghost || !stack) return;
  document.getElementById("ovCap").textContent = OV_CAPTIONS[m];
  if (m === "ghost") { ghost.style.display = ""; stack.style.display = "none"; fade.style.display = "none"; }
  else {
    ghost.style.display = "none"; stack.style.display = ""; syncStack();
    const top = document.getElementById("ovTop");
    if (m === "wipe") {
      fade.style.display = "none"; div.style.display = "";
      top.style.opacity = 1; applyWipe(parseFloat(div.style.left) || 50);
      document.getElementById("tagL").style.display = ""; document.getElementById("tagR").style.display = "";
    } else {
      fade.style.display = ""; div.style.display = "none";
      top.style.clipPath = "none"; top.style.opacity = fade.value / 100 * -1 + 1;
      document.getElementById("tagL").style.display = "none"; document.getElementById("tagR").style.display = "none";
    }
  }
}
function applyWipe(pct){
  const top = document.getElementById("ovTop"), div = document.getElementById("ovDiv");
  if (!top) return;
  pct = Math.max(2, Math.min(98, pct));
  top.style.opacity = 1;
  top.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;   // raw visible LEFT of divider
  div.style.left = pct + "%";
}
document.getElementById("ovModes").addEventListener("click", e => {
  const b = e.target.closest(".ovm"); if (b) setOvMode(b.dataset.m);
});
document.getElementById("ovFade").addEventListener("input", e => {
  const top = document.getElementById("ovTop");
  if (top) { top.style.clipPath = "none"; top.style.opacity = 1 - e.target.value / 100; }
});
document.addEventListener("pointerdown", e => {
  if (e.target.id !== "ovDiv") return;
  const stack = document.getElementById("ovStack");
  const move = ev => {
    const r = stack.getBoundingClientRect();
    applyWipe((ev.clientX - r.left) / r.width * 100);
  };
  const up = () => { document.removeEventListener("pointermove", move);
                     document.removeEventListener("pointerup", up); };
  document.addEventListener("pointermove", move);
  document.addEventListener("pointerup", up);
});
function syncStack(){
  const m = document.querySelector('#mjcBox video');
  const b = document.getElementById('ovBase'), t = document.getElementById('ovTop');
  if (m && b && t) { b.currentTime = m.currentTime; t.currentTime = m.currentTime; }
}
// play/pause/seek on any pane video mirrors to all pane videos
document.addEventListener('play', e => {
  if (e.target.tagName !== 'VIDEO') return;
  document.querySelectorAll('.stage video').forEach(v => { if (v !== e.target && v.paused) v.play().catch(()=>{}); });
}, true);
document.addEventListener('pause', e => {
  if (e.target.tagName !== 'VIDEO') return;
  document.querySelectorAll('.stage video').forEach(v => { if (v !== e.target && !v.paused) v.pause(); });
}, true);
document.addEventListener('seeked', e => {
  if (e.target.tagName !== 'VIDEO') return;
  const t = e.target.currentTime;
  document.querySelectorAll('.stage video').forEach(v => {
    if (v !== e.target && Math.abs(v.currentTime - t) > 0.08 && isFinite(v.duration))
      v.currentTime = Math.min(t, v.duration - 0.05);
  });
}, true);

// ---------- physics plots, drawn progressively with the video ----------
// The trace is NOT a finished curve with a cursor sweeping over it: the polyline
// itself grows one frame at a time from the video's currentTime, so the plot is
// generated as the motion plays. The full curve sits behind it at low opacity
// only so the vertical scale reads as fixed rather than jumping.
const CW=600, CH=100, PADL=38, PADR=10, PADT=15, PADB=16;
const FLOAT_TH=2.0, PEN_TH=-1.0;   // cm of tolerance either side of the floor
function lerp(a,b,t){return a+(b-a)*t}

function buildChart(holder, vals, fps, opts){
  const n=vals.length, dur=n/fps;
  const y0=opts.y0, y1=opts.y1;
  const X=t=>PADL+(CW-PADL-PADR)*(dur?t/dur:0), Y=v=>PADT+(CH-PADT-PADB)*(1-(v-y0)/(y1-y0));
  let z='';
  (opts.zones||[]).forEach(zn=>{
    const a=Math.max(y0,zn.from), b=Math.min(y1,zn.to);
    if(b>a) z+=`<rect x="${PADL}" y="${Y(b)}" width="${CW-PADL-PADR}" height="${Y(a)-Y(b)}" fill="${zn.color}" opacity="${zn.op}"></rect>
      <text x="${CW-PADR-4}" y="${Y(b)+10}" text-anchor="end" font-size="9" fill="var(--muted)">${zn.label}</text>`;
  });
  let grid='';
  for(let i=0;i<=3;i++){
    const v=lerp(y0,y1,i/3), yy=Y(v);
    grid+=`<line x1="${PADL}" x2="${CW-PADR}" y1="${yy}" y2="${yy}" stroke="var(--line)" stroke-width="0.6" opacity="0.7"></line>
           <text x="${PADL-5}" y="${yy+3}" text-anchor="end" font-size="9" fill="var(--muted)">${v.toFixed(0)}</text>`;
  }
  const pts=vals.map((v,i)=>`${X(i/fps).toFixed(1)},${Y(v).toFixed(1)}`);
  const zero=(y0<0&&y1>0)?`<line x1="${PADL}" x2="${CW-PADR}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--ink)" stroke-width="0.8" opacity="0.5"></line>`:'';
  holder.innerHTML=`<svg viewBox="0 0 ${CW} ${CH}" style="width:100%;display:block">
    <text x="${PADL}" y="10" font-size="10" fill="var(--muted)">${opts.title}</text>
    ${z}${grid}${zero}
    <polyline points="${pts.join(' ')}" fill="none" stroke="${opts.color}" stroke-width="1.3" opacity="0.13"></polyline>
    <polyline class="live" points="" fill="none" stroke="${opts.color}" stroke-width="2.1"
              stroke-linejoin="round" stroke-linecap="round"></polyline>
    <circle class="head" r="3.2" fill="${opts.color}" opacity="0"></circle>
    <line class="xhair" x1="${PADL}" x2="${PADL}" y1="${PADT}" y2="${CH-PADB}" stroke="var(--muted)" stroke-width="0.8" opacity="0" stroke-dasharray="3 3"></line>
    <rect class="hit" x="${PADL}" y="0" width="${CW-PADL-PADR}" height="${CH}" fill="transparent" style="cursor:crosshair"></rect>
  </svg>`;
  const svg=holder.firstElementChild, hit=svg.querySelector('.hit'),
        xh=svg.querySelector('.xhair'), tip=document.getElementById('chartTip'),
        live=svg.querySelector('.live'), head=svg.querySelector('.head');
  const toT=ev=>{const r=svg.getBoundingClientRect();
    const fx=(ev.clientX-r.left)/r.width*CW;
    return Math.max(0,Math.min(dur,(fx-PADL)/(CW-PADL-PADR)*dur));};
  hit.addEventListener('mousemove',ev=>{
    const t=toT(ev), i=Math.min(n-1,Math.round(t*fps));
    xh.setAttribute('x1',X(t)); xh.setAttribute('x2',X(t)); xh.setAttribute('opacity','1');
    tip.style.display='block'; tip.style.left=(ev.clientX+12)+'px'; tip.style.top=(ev.clientY-10)+'px';
    tip.textContent=`t=${t.toFixed(2)}s  ${opts.tipLabel}: ${vals[i].toFixed(1)} cm`;
  });
  hit.addEventListener('mouseleave',()=>{xh.setAttribute('opacity','0');tip.style.display='none';});
  hit.addEventListener('click',ev=>seekAll(toT(ev)));
  const c={_i:-1};
  c.setT=t=>{
    const i=Math.max(0,Math.min(n-1,Math.round(t*fps)));
    if(i===c._i) return;                 // redraw only when the frame actually advances
    c._i=i;
    live.setAttribute('points', pts.slice(0,i+1).join(' '));
    const xy=pts[i].split(',');
    head.setAttribute('cx',xy[0]); head.setAttribute('cy',xy[1]); head.setAttribute('opacity','1');
  };
  c.setT(0);
  return c;
}

// ---------- combined chart: raw and corrected on one set of axes ----------
// The Row 2 plots answer "what did each pass do". This one answers "what did the
// correction change", which you cannot read off two charts sitting side by side.
const OV_CH = 152;
function buildOverlayChart(holder, sets, opts){
  const fps = opts.fps;
  const n = Math.max(...sets.map(s => s.vals.length));
  const dur = n / fps, y0 = opts.y0, y1 = opts.y1;
  const X = t => PADL+(CW-PADL-PADR)*(dur?t/dur:0);
  const Y = v => PADT+(OV_CH-PADT-PADB)*(1-(v-y0)/(y1-y0));
  let z = '';
  (opts.zones||[]).forEach(zn => {
    const a = Math.max(y0, zn.from), b = Math.min(y1, zn.to);
    if(b > a) z += `<rect x="${PADL}" y="${Y(b)}" width="${CW-PADL-PADR}" height="${Y(a)-Y(b)}" fill="${zn.color}" opacity="${zn.op}"></rect>
      <text x="${CW-PADR-4}" y="${Y(b)+10}" text-anchor="end" font-size="9" fill="var(--muted)">${zn.label}</text>`;
  });
  let grid = '';
  for(let i=0;i<=4;i++){
    const v = lerp(y0,y1,i/4), yy = Y(v);
    grid += `<line x1="${PADL}" x2="${CW-PADR}" y1="${yy}" y2="${yy}" stroke="var(--line)" stroke-width="0.6" opacity="0.7"></line>
             <text x="${PADL-5}" y="${yy+3}" text-anchor="end" font-size="9" fill="var(--muted)">${v.toFixed(0)}</text>`;
  }
  const zero = (y0<0&&y1>0) ? `<line x1="${PADL}" x2="${CW-PADR}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--ink)" stroke-width="0.8" opacity="0.55"></line>` : '';
  const prepared = sets.map(s => ({
    ...s, pts: s.vals.map((v,i) => `${X(i/fps).toFixed(1)},${Y(v).toFixed(1)}`)
  }));
  let lines = '', key = '';
  prepared.forEach((s,i) => {
    lines += `<polyline points="${s.pts.join(' ')}" fill="none" stroke="${s.color}" stroke-width="1.3" opacity="0.13"></polyline>
      <polyline class="live${i}" points="" fill="none" stroke="${s.color}" stroke-width="2.2"
                stroke-linejoin="round" stroke-linecap="round"></polyline>
      <circle class="head${i}" r="3.4" fill="${s.color}" opacity="0"></circle>`;
    key += `<rect x="${PADL+i*136}" y="${OV_CH-9}" width="9" height="3" fill="${s.color}"></rect>
      <text x="${PADL+i*136+13}" y="${OV_CH-6}" font-size="9.5" fill="var(--muted)">${s.label}</text>`;
  });
  holder.innerHTML = `<svg viewBox="0 0 ${CW} ${OV_CH}" style="width:100%;display:block">
    <text x="${PADL}" y="10" font-size="10" fill="var(--muted)">${opts.title}</text>
    ${z}${grid}${zero}${lines}${key}
    <line class="xhair" x1="${PADL}" x2="${PADL}" y1="${PADT}" y2="${OV_CH-PADB}" stroke="var(--muted)" stroke-width="0.8" opacity="0" stroke-dasharray="3 3"></line>
    <rect class="hit" x="${PADL}" y="0" width="${CW-PADL-PADR}" height="${OV_CH-12}" fill="transparent" style="cursor:crosshair"></rect>
  </svg>`;
  const svg = holder.firstElementChild, hit = svg.querySelector('.hit'),
        xh = svg.querySelector('.xhair'), tip = document.getElementById('chartTip');
  const live = prepared.map((_,i) => svg.querySelector('.live'+i));
  const head = prepared.map((_,i) => svg.querySelector('.head'+i));
  const toT = ev => { const r = svg.getBoundingClientRect();
    const fx = (ev.clientX-r.left)/r.width*CW;
    return Math.max(0, Math.min(dur, (fx-PADL)/(CW-PADL-PADR)*dur)); };
  hit.addEventListener('mousemove', ev => {
    const t = toT(ev);
    xh.setAttribute('x1',X(t)); xh.setAttribute('x2',X(t)); xh.setAttribute('opacity','1');
    const parts = prepared.map(s => {
      const i = Math.min(s.vals.length-1, Math.round(t*fps));
      return `${s.label} ${s.vals[i].toFixed(1)}`;
    });
    tip.style.display='block'; tip.style.left=(ev.clientX+12)+'px'; tip.style.top=(ev.clientY-10)+'px';
    tip.textContent = `t=${t.toFixed(2)}s  ${parts.join('   ')} ${opts.unit||''}`;
  });
  hit.addEventListener('mouseleave', () => { xh.setAttribute('opacity','0'); tip.style.display='none'; });
  hit.addEventListener('click', ev => seekAll(toT(ev)));
  const c = {_i:-1};
  c.setT = t => {
    const gi = Math.round(t*fps);
    if(gi === c._i) return;
    c._i = gi;
    prepared.forEach((s,k) => {
      const i = Math.max(0, Math.min(s.pts.length-1, gi));
      live[k].setAttribute('points', s.pts.slice(0,i+1).join(' '));
      const xy = s.pts[i].split(',');
      head[k].setAttribute('cx',xy[0]); head[k].setAttribute('cy',xy[1]);
      head[k].setAttribute('opacity','1');
    });
  };
  c.setT(0);
  return c;
}

// ---------- contact ribbons for the overlay row ----------
function contactClass(v){ return v<PEN_TH ? 'pen' : (v>FLOAT_TH ? 'float' : 'ok'); }
const RIB={ok:'var(--mk-corr)', float:'var(--mk-float)', pen:'var(--mk-pen)'};
function addRibbon(parent, ser, fpsFallback, label, labColor){
  const vals=ser.foot_cm, n=vals.length, fps=ser.fps||fpsFallback, dur=n/fps, W=1000, H=13;
  const runs=[]; let cur=null;
  vals.forEach((v,i)=>{ const k=contactClass(v);
    if(!cur||cur.k!==k){ cur={k,a:i,b:i}; runs.push(cur); } else cur.b=i; });
  const rects=runs.map(r=>{
    const x=r.a/n*W, w=Math.max(0.6,(r.b-r.a+1)/n*W);
    return `<rect x="${x.toFixed(2)}" y="0" width="${w.toFixed(2)}" height="${H}" fill="${RIB[r.k]}"></rect>`;
  }).join('');
  const pf=vals.filter(v=>v>FLOAT_TH).length/n*100, pp=vals.filter(v=>v<PEN_TH).length/n*100;
  const box=document.createElement('div');
  box.className='ribbon';
  box.innerHTML=`<div class="rlab" style="color:${labColor}">${label}
      <span style="color:var(--muted);font-weight:400">${(100-pf-pp).toFixed(0)}% in contact,
      ${pf.toFixed(0)}% floating, ${pp.toFixed(0)}% penetrating</span></div>
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"
         style="width:100%;height:13px;display:block;border-radius:3px;border:1px solid var(--line)">
      ${rects}<rect class="play" x="0" y="0" width="3" height="${H}" fill="var(--ink)" opacity="0"></rect>
    </svg>`;
  parent.appendChild(box);
  const play=box.querySelector('.play');
  return {setT:t=>{ play.setAttribute('x',((dur?Math.min(1,t/dur):0)*W).toFixed(1));
                    play.setAttribute('opacity','0.85'); }};
}

// ---------- wiring ----------
let tracked=[];          // {v: <video>, parts:[{setT}]}
let raf=null;
function seekAll(t){
  document.querySelectorAll('.stage video').forEach(v=>{
    if(isFinite(v.duration)) v.currentTime=Math.min(t, v.duration-0.05);
  });
}
function startLoop(){
  if(raf) return;
  const step=()=>{
    tracked.forEach(p=>{ if(p.v && isFinite(p.v.duration))
      p.parts.forEach(c=>c.setT(p.v.currentTime)); });
    raf=requestAnimationFrame(step);
  };
  raf=requestAnimationFrame(step);
}
// Raw and corrected share one vertical scale per metric, otherwise a floating raw
// motion and a grounded corrected one draw as the same picture.
function domain(a, b, lo, hi){
  const all=[].concat(a||[], b||[]);
  const mn=Math.min(lo, ...all), mx=Math.max(hi, ...all);
  const pad=(mx-mn)*0.08+0.4;
  return [mn-pad, mx+pad];
}
function renderCharts(d){
  tracked=[];
  const pr=document.getElementById('plotsRaw'), pc=document.getElementById('plotsCorr'),
        rib=document.getElementById('ovRibbons'), img=document.getElementById('physPlot');
  pr.innerHTML=''; pc.innerHTML=''; rib.innerHTML='';
  const S=d.series, B=d.series_before;
  if(!S && !B){
    pr.innerHTML=pc.innerHTML='<div class="ptitle">No physics trace for this motion.</div>';
    if(d.diag_plot){ img.src=d.diag_plot; img.style.display=''; } else img.style.display='none';
    return;
  }
  img.style.display='none';
  const fps=(S&&S.fps)||(B&&B.fps)||30;
  const [f0,f1]=domain(B&&B.foot_cm, S&&S.foot_cm, -2, 6);
  const [c0,c1]=domain(B&&B.com_cm,  S&&S.com_cm,   0, 0);
  const footZones=[{from:-999,to:PEN_TH,color:'var(--mk-pen)',op:0.18,label:'penetrating'},
                   {from:FLOAT_TH,to:999,color:'var(--mk-float)',op:0.18,label:'floating'}];
  const fill=(holder, ser, color)=>{
    if(!ser){ holder.innerHTML='<div class="ptitle">No trace for this pass.</div>'; return null; }
    const h1=document.createElement('div'), h2=document.createElement('div');
    h2.style.marginTop='5px';
    holder.appendChild(h1); holder.appendChild(h2);
    return [
      buildChart(h1, ser.foot_cm, ser.fps||fps, {title:'Lowest foot above floor (cm), 0 = ground',
        tipLabel:'foot', color, y0:f0, y1:f1, zones:footZones}),
      buildChart(h2, ser.com_cm, ser.fps||fps, {title:'CoM offset from feet centroid (cm)',
        tipLabel:'CoM offset', color, y0:c0, y1:c1}),
    ];
  };
  const cRaw=fill(pr, B, 'var(--mk-raw)'), cCor=fill(pc, S, 'var(--mk-corr)');
  const vRaw=document.querySelector('#robotBox video'),
        vCor=document.querySelector('#mjcBox video'),
        vOv =document.querySelector('#ovBox video');
  if(cRaw) tracked.push({v:vRaw||vCor||vOv, parts:cRaw});
  if(cCor) tracked.push({v:vCor||vRaw||vOv, parts:cCor});
  // Combined overlay plots: both passes on one set of axes, so the correction
  // reads as the gap between the red and green traces rather than as something
  // you have to hold in your head while looking back and forth across Row 2.
  const combo = document.getElementById('ovCombo');
  combo.innerHTML = '';
  const comboParts = [];
  if(B && S){
    const mk = (metric, title, y0, y1, zones) => {
      const h = document.createElement('div');
      h.style.marginTop = '6px';
      combo.appendChild(h);
      return buildOverlayChart(h, [
        {vals:B[metric], color:'var(--mk-raw)',  label:'RAW (AI generated)'},
        {vals:S[metric], color:'var(--mk-corr)', label:'CORRECTED (Pragya)'},
      ], {title, fps, y0, y1, zones, unit:'cm'});
    };
    comboParts.push(mk('foot_cm', 'Overlay: lowest foot above floor (cm), raw vs corrected, 0 = ground',
                       f0, f1, footZones));
    comboParts.push(mk('com_cm', 'Overlay: CoM offset from feet centroid (cm), raw vs corrected',
                       c0, c1, []));
  } else {
    combo.innerHTML = '<div class="ptitle">Combined plot needs both the raw and the corrected trace.</div>';
  }

  const ribParts=[];
  if(B) ribParts.push(addRibbon(rib, B, fps, 'RAW: AI generated',  'var(--mk-raw)'));
  if(S) ribParts.push(addRibbon(rib, S, fps, 'CORRECTED: Pragya', 'var(--mk-corr)'));
  const row3 = comboParts.concat(ribParts);
  if(row3.length) tracked.push({v:vOv||vCor||vRaw, parts:row3});
  startLoop();
}

show();

/* ================= Intro trailer controls ================= */
(function(){
  const v = document.getElementById("introVid");
  if(!v) return;
  // If the trailer asset is not deployed yet, remove the hero instead of showing
  // a black box above the review.
  const die = () => { const w = document.getElementById("hero"); if(w) w.remove(); };
  v.addEventListener("error", die);
  const src = v.querySelector("source");
  if(src) src.addEventListener("error", die);
  const pb = document.getElementById("introPause");
  pb.onclick = () => {
    if(v.paused){ v.play(); pb.innerHTML = "&#10074;&#10074; Pause"; }
    else { v.pause(); pb.innerHTML = "&#9654; Play"; }
  };
  document.getElementById("introFS").onclick = () => {
    if(v.requestFullscreen) v.requestFullscreen();
    else if(v.webkitEnterFullscreen) v.webkitEnterFullscreen();
  };
})();

/* ================= Interactive training floor ================= */
// A vector 3D replica of the arena the trailer is filmed in: same three training
// boxes at the same world coordinates (build_army_scene.ZONES), same staggered
// 5x2 formation per box (render_army_trailer.formation). Resolution-independent
// CSS/SVG, so it is crisp on any display density.
const ZONES3D = [
  {key:"beginner", label:"BEGINNER", cx:-12, tint:"rgba(96,140,210,.16)",
   desc:"Stances · balance · footwork fundamentals"},
  {key:"defense",  label:"DEFENSE",  cx:0,   tint:"rgba(60,170,120,.15)",
   desc:"Blocks · guards · evasion & counters"},
  {key:"attack",   label:"ATTACK",   cx:12,  tint:"rgba(220,110,70,.15)",
   desc:"Kicks · strikes · leaps & charges"},
];
const FLOOR = { S:20, W:880, H:340, HX:4.6, HY:3.6 };   // px per metre, plane size, zone half-extents

const BOT_POSES = [
  '<path d="M8 6.4 8 14M8 8 3.6 11.6M8 8 12.4 11.2M8 14 4.8 22M8 14 11.4 22"/>',
  '<path d="M8 6.4 8.4 14M8 8 4 6.2M8 8 12.6 9.8M8.4 14 4 20.4M8.4 14 13.4 16.6"/>',
  '<path d="M8 6.4 7.6 14M8 8 3.2 8.8M8 8 12.8 6.6M7.6 14 4.2 21.6M7.6 14 12.2 20.2"/>',
];
function botSVG(pose){
  return `<svg viewBox="0 0 16 26"><g fill="none" stroke="#4a5262" stroke-width="2.1"
    stroke-linecap="round"><circle cx="8" cy="3.4" r="2.5" fill="#eef0f4" stroke-width="1.5"/>
    ${BOT_POSES[pose % BOT_POSES.length]}</g></svg>`;
}

function setZone(z){
  navZone = (navZone === z) ? "" : z;
  document.querySelectorAll(".zone3d,.zcard").forEach(el =>
    el.classList.toggle("picked", !!navZone && el.dataset.zone === navZone));
  renderNav();
  if(navZone) document.querySelector(".wrap").scrollIntoView({behavior:"smooth"});
}

(function buildFloor(){
  const plane = document.getElementById("floorPlane");
  const stage = document.getElementById("floorStage");
  if(!plane || !stage) return;
  const {S, W, H, HX, HY} = FLOOR;
  const px = (x,y) => [W/2 + x*S, H/2 - y*S];
  const counts = {};
  DATA.forEach(d => { const z = zoneOf(d); counts[z] = (counts[z]||0) + 1; });

  ZONES3D.forEach((z, zi) => {
    const [lx, ty] = px(z.cx - HX, HY);
    const zd = document.createElement("div");
    zd.className = "zone3d"; zd.dataset.zone = z.key;
    zd.style.cssText = `left:${lx}px;top:${ty}px;width:${2*HX*S}px;height:${2*HY*S}px;
      background:${z.tint}`;
    zd.innerHTML = `<span class="zlabel">${z.label}</span>`;
    // corner ticks, matching the rendered arena's inset marks
    [[0,0],[0,1],[1,0],[1,1]].forEach(([cx2,cy2]) => {
      const t = document.createElement("span"); t.className = "tick";
      t.style.cssText = `width:30px;height:3px;${cx2?"right":"left"}:8px;${cy2?"bottom":"top"}:6px`;
      zd.appendChild(t);
    });
    zd.addEventListener("click", () => setZone(z.key));
    zd.addEventListener("pointerenter", () => zd.classList.add("hot"));
    zd.addEventListener("pointerleave", () => { zd.classList.remove("hot"); tipHide(); });
    zd.addEventListener("pointermove", ev => tipShow(ev, z, counts[z.key]||0));
    plane.appendChild(zd);

    for(let i = 0; i < 10; i++){                    // the trailer's 5x2 ranks
      const r = Math.floor(i/5), c = i % 5;
      const wx = z.cx + (c - 2)*1.85 + (r ? 0.92 : 0);
      const wy = -1.55 + r*3.10;
      const [bx, by] = px(wx, wy);
      const b = document.createElement("div");
      b.className = "bot";
      b.style.cssText = `left:${bx}px;top:${by}px`;
      b.innerHTML = `<span class="shadow"></span><span class="fig">${botSVG(zi*3 + i)}</span>`;
      plane.appendChild(b);
    }
  });

  // zone cards under the floor
  document.getElementById("zoneCards").innerHTML = ZONES3D.map(z =>
    `<div class="zcard" data-zone="${z.key}" onclick="setZone('${z.key}')">
       <span class="cnt">${counts[z.key]||0} moves</span><b>${z.label} BOX</b>
       <div class="zn">${z.desc} · 10 humanoids</div></div>`).join("");

  // orbit drag
  const HOME = {rx:57, rz:-24};
  let rx = HOME.rx, rz = HOME.rz, drag = null;
  const apply = () => {
    plane.style.setProperty("--rx", rx + "deg");
    plane.style.setProperty("--rz", rz + "deg");
  };
  stage.addEventListener("pointerdown", ev => {
    drag = {x:ev.clientX, y:ev.clientY, rx, rz};
    stage.classList.add("grabbing");
    stage.setPointerCapture(ev.pointerId);
  });
  stage.addEventListener("pointermove", ev => {
    if(!drag) return;
    rz = drag.rz + (ev.clientX - drag.x) * 0.25;
    rx = Math.min(82, Math.max(20, drag.rx - (ev.clientY - drag.y) * 0.25));
    rz = Math.min(60, Math.max(-100, rz));
    apply();
  });
  const end = () => { drag = null; stage.classList.remove("grabbing"); };
  stage.addEventListener("pointerup", end);
  stage.addEventListener("pointercancel", end);
  document.getElementById("floorReset").onclick = () => { rx = HOME.rx; rz = HOME.rz; apply(); };

  const tip = document.getElementById("zoneTip");
  function tipShow(ev, z, n){
    tip.style.display = "block";
    tip.innerHTML = `<b>${z.label} BOX</b><div class="zn">${z.desc}</div>
      <div class="zn">${n} library moves · click to filter</div>`;
    tip.style.left = Math.min(window.innerWidth - 250, ev.clientX + 16) + "px";
    tip.style.top = (ev.clientY + 14) + "px";
  }
  function tipHide(){ tip.style.display = "none"; }
})();
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="results_review/render_manifest.json")
    ap.add_argument("--human-dir", default="data/kalari_videos",
                    help="directory of original human videos, named <motion_id>.mp4")
    ap.add_argument("--out", default="results_review/review.html")
    ap.add_argument("--title", default="KalariSena: Human vs G1 Retarget Review")
    ap.add_argument("--only", default=None,
                    help="comma-separated motion ids: page shows ONLY these")
    ap.add_argument("--require-human", action="store_true",
                    help="omit robot motions that have no matching human video")
    ap.add_argument("--exclude", default=None,
                    help="comma-separated motion ids to keep OUT of the page, on top "
                         f"of the built-in list ({', '.join(sorted(EXCLUDE_MOTIONS))})")
    ap.add_argument("--diagnostics", default="results_review/diagnostics.json",
                    help="physics diagnostics from scripts/motion_diagnostics.py")
    ap.add_argument("--sources", default="data/kalari_sources.csv",
                    help="clip spec CSV; supplies the readable move name and source "
                         "troupe shown in the menu")
    args = ap.parse_args()

    if not os.path.isfile(args.manifest):
        raise SystemExit(
            f"manifest not found: {args.manifest}\n"
            f"Run scripts/render_g1_motion.py first to produce it."
        )
    with open(args.manifest) as fh:
        records = json.load(fh)

    human_map = find_human_videos(args.human_dir)
    if args.only:
        keep = {m.strip() for m in args.only.split(",") if m.strip()}
        records = [r for r in records if r["motion_id"] in keep]
        human_map = {k: v for k, v in human_map.items() if k in keep}
        print(f"--only filter: {len(keep)} ids -> {len(records)} records, {len(human_map)} human videos")

    drop = set(EXCLUDE_MOTIONS)
    if args.exclude:
        drop |= {m.strip() for m in args.exclude.split(",") if m.strip()}
    if drop:
        hit = sorted(m for m in drop
                     if any(r["motion_id"] == m for r in records) or m in human_map)
        records = [r for r in records if r["motion_id"] not in drop]
        human_map = {k: v for k, v in human_map.items() if k not in drop}
        if hit:
            print(f"excluded {len(hit)} motion(s) from the published review: {hit}")
        missing = sorted(drop - set(hit))
        if missing:
            print(f"[warn] excluded id(s) not present in this build: {missing}")
    diag = {}
    if os.path.isfile(args.diagnostics):
        with open(args.diagnostics) as fh:
            diag = {d["motion_id"]: d for d in json.load(fh)}
        print(f"physics diagnostics loaded for {len(diag)} motion(s)")
    cmp_path = os.path.join(os.path.dirname(args.diagnostics) or ".", "compare_manifest.json")
    cmp_map = {}
    if os.path.isfile(cmp_path):
        with open(cmp_path) as fh:
            cmp_map = {d["motion_id"]: d for d in json.load(fh)}
        print(f"compare renders loaded for {len(cmp_map)} motion(s)")
    rl_path = os.path.join(os.path.dirname(args.diagnostics) or ".", "rl_manifest.json")
    rl = {}
    if os.path.isfile(rl_path):
        with open(rl_path) as fh:
            rl = {d["motion_id"]: d for d in json.load(fh)}
        print(f"RL results loaded for {len(rl)} motion(s)")
    before_path = args.diagnostics.replace(".json", "_before.json")
    if os.path.isfile(before_path):
        with open(before_path) as fh:
            for d in json.load(fh):
                if d["motion_id"] in diag:
                    diag[d["motion_id"]]["before"] = d
        print("before/after correction comparison enabled")
    names, srcs, quals = load_sources(args.sources)
    if names:
        print(f"move names loaded for {len(names)} motion(s) from {args.sources}")
    else:
        print(f"[warn] no move names found in {args.sources} -- the menu will fall "
              f"back to names derived from the motion ids")
    named = sum(1 for r in records if r["motion_id"] in names)
    if named < len(records):
        unnamed = [r["motion_id"] for r in records if r["motion_id"] not in names]
        print(f"[warn] {len(unnamed)} motion(s) have no curated name, using derived: {unnamed}")

    path, paired = build(records, human_map, args.out, args.title,
                         require_human=args.require_human, diagnostics=diag, rl=rl,
                         cmp_map=cmp_map, names=names, srcs=srcs, quals=quals)

    print(f"motions in manifest : {len(records)}")
    print(f"human videos found  : {len(human_map)} in {args.human_dir}"
          f"{'  (directory does not exist)' if not os.path.isdir(args.human_dir) else ''}")
    print(f"paired side-by-side : {paired}/{len(records)}")
    unpaired = [r['motion_id'] for r in records
                if r['motion_id'] not in human_map][:8]
    if unpaired:
        print(f"unpaired (first few): {unpaired}")
    print(f"\nwrote {path}")
    print(f"open it with:  open {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
