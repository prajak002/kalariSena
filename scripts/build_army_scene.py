#!/usr/bin/env python3
"""Generate a MuJoCo scene holding N copies of the existing Unitree G1.

Writes assets/unitree_g1/army_<n>.xml, which reuses the official G1 MJCF verbatim:
the same mesh assets, the same body/joint hierarchy, the same proportions. Nothing
about the robot is redesigned; the subtree is copied and its names are prefixed.

Why copy rather than <replicate>: replicate applies a fixed positional increment per
copy, which can only lay out a regular lattice. Each robot here needs its own place
and its own facing, and every copy carries a free joint, so poses are written
straight into qpos at render time and the XML only has to provide N addressable
robots.

Actuators and sensors are dropped. This scene is posed kinematically (mj_forward on
qpos); leaving 29 actuators per robot would build a 900-column control space that is
never written to, and the IMU sensors reference sites we would then have to keep
unique for no benefit.

Usage
    python3 scripts/build_army_scene.py --count 30
"""

from __future__ import annotations

import argparse
import copy
import os
import xml.etree.ElementTree as ET

SRC = "assets/unitree_g1/g1_29dof_rev_1_0.xml"
RENAME_TAGS = ("body", "joint", "site", "camera", "geom")


def prefix_names(elem: ET.Element, prefix: str) -> None:
    """Prefix every name attribute in a subtree so N copies can coexist."""
    for e in elem.iter():
        if e.tag in RENAME_TAGS and e.get("name"):
            e.set("name", f"{prefix}{e.get('name')}")


def cinematic_visual() -> ET.Element:
    """Render settings for the arena look.

    MuJoCo's rasteriser gives us shadow maps, multisampling and horizon haze. It has
    no volumetrics and no depth of field, so those are approximated afterwards from
    the depth buffer in the post pass; see render_army_trailer.py.

    This is a bright facility, not a night battlefield: ambient is lifted hard so the
    white floor stays white in shadow instead of going grey, which is what separates
    "premium showroom" from "dark arena with a pale floor".
    """
    vis = ET.Element("visual")
    ET.SubElement(vis, "headlight", diffuse="0.42 0.43 0.46",
                  ambient="0.30 0.31 0.34", specular="0.30 0.30 0.32")
    ET.SubElement(vis, "rgba", haze="0.88 0.90 0.94 1")
    # 16k shadow maps: at 4K delivery the 8k map's texels are visible as soft
    # stair-stepping along contact shadows under each robot.
    ET.SubElement(vis, "quality", shadowsize="16384", offsamples="8")
    ET.SubElement(vis, "map", fogstart="8", fogend="55", znear="0.05", zfar="120",
                  shadowclip="6", shadowscale="0.8")
    # Offscreen buffer sized for 1.5x supersampling of a 3840x2160 delivery.
    ET.SubElement(vis, "global", azimuth="-140", elevation="-20", offwidth="5760",
                  offheight="3240")
    return vis


def cinematic_assets() -> ET.Element:
    """White polished arena floor, zone surfaces, and the sponsor banner textures."""
    a = ET.Element("asset")
    ET.SubElement(a, "texture", type="skybox", builtin="gradient",
                  rgb1="0.90 0.92 0.96", rgb2="0.74 0.78 0.85",
                  width="512", height="3072")
    # Flat white, not a checker: any tiling pattern at this scale reads as a garage
    # floor. Reflectance carries the polish instead of texture.
    ET.SubElement(a, "texture", type="2d", name="arena_floor", builtin="flat",
                  rgb1="0.955 0.960 0.968", rgb2="0.955 0.960 0.968",
                  width="128", height="128")
    ET.SubElement(a, "material", name="arena_floor", texture="arena_floor",
                  texuniform="true", texrepeat="1 1", reflectance="0.42",
                  specular="0.55", shininess="0.82")
    # Zone surfaces: barely-there tints so the three areas separate without turning
    # the floor into a colour block and losing "white floor dominant".
    for nm, rgba in (("zone_beginner", "0.86 0.90 0.96 1"),
                     ("zone_defense",  "0.87 0.93 0.90 1"),
                     ("zone_attack",   "0.96 0.89 0.86 1")):
        ET.SubElement(a, "material", name=nm, rgba=rgba, reflectance="0.30",
                      specular="0.45", shininess="0.70")
    ET.SubElement(a, "material", name="zone_line", rgba="0.16 0.20 0.28 1",
                  reflectance="0.10", specular="0.25", shininess="0.4")
    ET.SubElement(a, "material", name="barrier", rgba="0.90 0.91 0.94 1",
                  reflectance="0.22", specular="0.40", shininess="0.6")
    for nm in ("kalarisena_hero", "pragya_velvet"):
        ET.SubElement(a, "texture", type="2d", name=nm, file=f"banners/{nm}.png")
        # texuniform=false so the artwork maps once across the board face rather
        # than tiling, which is what keeps the logos intact and undistorted.
        ET.SubElement(a, "material", name=nm, texture=nm, texuniform="false",
                      texrepeat="1 1", specular="0.18", shininess="0.25",
                      reflectance="0.05")
    return a


# Zones run left to right along +x, in the order the camera tours them.
ZONES = [
    {"key": "beginner", "label": "BEGINNER", "cx": -12.0, "mat": "zone_beginner"},
    {"key": "defense",  "label": "DEFENSE",  "cx":   0.0, "mat": "zone_defense"},
    {"key": "attack",   "label": "ATTACK",   "cx":  12.0, "mat": "zone_attack"},
]
ZONE_HALF_X, ZONE_HALF_Y = 4.6, 3.6      # each training box, metres
BANNER_H = 1.85                          # common board height; width follows aspect
BANNER_ASPECT = {"kalarisena_hero": 1920 / 1080, "pragya_velvet": 1632 / 640}


def zone_markings() -> list[ET.Element]:
    """Three training boxes: tinted surface, inset border, corner ticks.

    Built from thin boxes stacked just above the floor plane rather than from a
    texture, so the lines stay crisp at any camera distance instead of turning to
    mush when the camera pushes in for a close-up.
    """
    out, t = [], 0.004
    for z in ZONES:
        cx = z["cx"]
        for (sx, sy, px, py) in ((ZONE_HALF_X, 0.05, 0, ZONE_HALF_Y),
                                 (ZONE_HALF_X, 0.05, 0, -ZONE_HALF_Y),
                                 (0.05, ZONE_HALF_Y, ZONE_HALF_X, 0),
                                 (0.05, ZONE_HALF_Y, -ZONE_HALF_X, 0)):
            out.append(ET.Element("geom", type="box", material="zone_line",
                                  pos=f"{cx + px} {py} {t * 2}",
                                  size=f"{sx} {sy} {t * 0.8}", contype="0", conaffinity="0"))
        for ox in (-1, 1):                       # corner ticks, inset
            for oy in (-1, 1):
                out.append(ET.Element(
                    "geom", type="box", material="zone_line",
                    pos=f"{cx + ox * (ZONE_HALF_X - 0.75)} {oy * (ZONE_HALF_Y - 0.05)} {t*2}",
                    size=f"0.75 0.035 {t}", contype="0", conaffinity="0"))
    return out


def banners() -> list[ET.Element]:
    """Sponsor boards on the arena barrier, alternating the two supplied artworks.

    Planes, not boxes. A 2D texture on a box gets swept along one axis and the
    artwork comes out as vertical smears; a plane maps the image once across its
    own extent, which is what keeps the logos intact.

    Board width is derived from each image's own aspect at a shared height, so
    neither artwork is stretched. Boards sit at stadium height, below the robots'
    heads, so they never mask a humanoid from the elevated eye.
    """
    out = []
    order = ["kalarisena_hero", "pragya_velvet"]
    back_y = ZONE_HALF_Y + 4.4
    # quats: rotate the plane's +z normal to face the arena interior
    FACE_S = "0.7071 0.7071 0 0"      # normal -> -y  (back wall, faces the camera)
    # local x -> along the wall, local y -> world up, normal -> inward.
    FACE_E = "0.5 0.5 0.5 0.5"        # normal -> +x  (left wall)
    FACE_W = "0.5 0.5 -0.5 -0.5"      # normal -> -x  (right wall)

    x, i = -21.0, 0
    while x < 21.0:                                        # back run
        nm = order[i % 2]
        w = BANNER_H * BANNER_ASPECT[nm] / 2.0
        out.append(ET.Element("geom", type="plane", material=nm, quat=FACE_S,
                              pos=f"{x + w} {back_y} {BANNER_H / 2}",
                              size=f"{w} {BANNER_H / 2} 0.1",
                              contype="0", conaffinity="0"))
        x += 2 * w + 0.30
        i += 1
    for sx, q in ((-21.5, FACE_E), (21.5, FACE_W)):        # side runs
        y, i = -11.0, 0
        while y < 7.0:
            nm = order[(i + 1) % 2]
            w = BANNER_H * BANNER_ASPECT[nm] / 2.0
            out.append(ET.Element("geom", type="plane", material=nm, quat=q,
                                  pos=f"{sx} {y + w} {BANNER_H / 2}",
                                  size=f"{w} {BANNER_H / 2} 0.1",
                                  contype="0", conaffinity="0"))
            y += 2 * w + 0.30
            i += 1
    # low barrier the boards sit on, so they read as built in rather than floating
    out.append(ET.Element("geom", type="box", material="barrier",
                          pos=f"0 {back_y + 0.12} {BANNER_H / 2}",
                          size=f"21.5 0.08 {BANNER_H / 2 + 0.05}",
                          contype="0", conaffinity="0"))
    for sx in (-21.6, 21.6):
        out.append(ET.Element("geom", type="box", material="barrier",
                              pos=f"{sx} -2.0 {BANNER_H / 2}",
                              size=f"0.08 9.2 {BANNER_H / 2 + 0.05}",
                              contype="0", conaffinity="0"))
    return out


def lighting() -> list[ET.Element]:
    """Key, cool fill and overhead wash for a bright indoor facility.

    White G1 shells on a white floor have almost no natural separation, so the key
    is strong and shadow-casting while ambient stays moderate. Flat, shadowless
    lighting was the first attempt and the robots dissolved into the floor.
    """
    out = []
    out.append(ET.Element("light", pos="-10 -16 15", dir="0.40 0.60 -0.70",
                          directional="true", castshadow="true",
                          diffuse="0.78 0.77 0.75", specular="0.34 0.34 0.36"))
    out.append(ET.Element("light", pos="12 18 12", dir="-0.45 -0.68 -0.58",
                          directional="true", castshadow="false",
                          diffuse="0.34 0.36 0.42", specular="0.40 0.42 0.50"))
    out.append(ET.Element("light", pos="0 0 20", dir="0 0 -1", directional="true",
                          castshadow="false", diffuse="0.26 0.26 0.28",
                          specular="0.10 0.10 0.11"))
    return out


def build(count: int, src: str, out_path: str) -> str:
    tree = ET.parse(src)
    root = tree.getroot()
    pelvis = next(b for b in root.find("worldbody").iter("body")
                  if b.get("name") == "pelvis")

    new = ET.Element("mujoco", model=f"g1_army_{count}")
    comp = root.find("compiler")
    ET.SubElement(new, "compiler", angle=comp.get("angle", "radian"),
                  meshdir=comp.get("meshdir", "meshes"), texturedir=".")
    # meshes are declared once and shared by every copy: 30 robots cost one upload
    for a in root.findall("asset"):
        new.append(copy.deepcopy(a))
    new.append(cinematic_assets())
    new.append(cinematic_visual())
    ET.SubElement(new, "statistic", center="0 0 1.0", extent="22")

    wb = ET.SubElement(new, "worldbody")
    for lt in lighting():
        wb.append(lt)
    ET.SubElement(wb, "geom", name="floor", size="0 0 0.05", type="plane",
                  material="arena_floor")
    for g in zone_markings():
        wb.append(g)
    for g in banners():
        wb.append(g)
    for i in range(count):
        b = copy.deepcopy(pelvis)
        prefix_names(b, f"r{i:02d}_")
        b.set("pos", "0 0 0.793")          # real placement is written into qpos
        wb.append(b)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ET.indent(tree, space="  ")
    ET.ElementTree(new).write(out_path, encoding="unicode", xml_declaration=False)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"assets/unitree_g1/army_{args.count}.xml"
    path = build(args.count, args.src, out)

    import mujoco
    m = mujoco.MjModel.from_xml_path(path)
    print(f"wrote {path}")
    print(f"  robots {args.count} | bodies {m.nbody} | geoms {m.ngeom} "
          f"| qpos {m.nq} | dof {m.nv} | lights {m.nlight}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
