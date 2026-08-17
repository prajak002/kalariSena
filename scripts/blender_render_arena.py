#!/usr/bin/env python3
"""Render the exported arena in Blender/Cycles.

Either through a Blender install:

    blender -b -P scripts/blender_render_arena.py -- \
        --export export_cycles --assets assets/unitree_g1 --out frames/ \
        --samples 128 --width 1920 --height 1080

or through Blender-as-a-module (`pip install bpy`), which needs no `--` and is
how this runs on a machine with no Blender app -- a Mac, for instance:

    .venv/bin/python scripts/blender_render_arena.py \
        --export export_cycles30 --assets assets/unitree_g1 --out frames/ \
        --device METAL

Consumes what export_cycles_scene.py baked, so the geometry, poses and camera are
exactly what MuJoCo computed; Cycles only changes how it is shaded.

Two things matter for this to survive a 1440-frame render:

  * mesh data is shared. 1770 instances reference ~64 unique meshes, so the scene
    costs one copy of the G1, not thirty.
  * transforms are driven by a frame handler reading the baked arrays, not by
    keyframes. Keyframing 1770 objects across 1440 frames is ~18M keyframes and
    Blender will not survive building that.
"""

import json
import os
import sys

import numpy as np

import bpy
import mathutils


def script_argv():
    """Blender swallows everything before `--`; the bpy module does not use one."""
    return (sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv
            else sys.argv[1:])


def parse():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--assets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--device", default="OPTIX",
                    choices=["OPTIX", "CUDA", "HIP", "METAL", "CPU"])
    ap.add_argument("--bounces", type=int, default=8,
                    help="light bounces. A white room needs far fewer than an\n"
                         "interior with dark corners; 4 is visually identical here\n"
                         "and materially cheaper.")
    ap.add_argument("--nodof", action="store_true", help="disable depth of field")
    ap.add_argument("--lightscale", type=float, default=1.0,
                    help="multiply every light and the world background. The set is\n"
                         "a white floor under white robots, which blows out long\n"
                         "before a normal interior would; this is the knob to pull.")
    ap.add_argument("--exposure", type=float, default=0.0,
                    help="stops of exposure applied by the view transform")
    ap.add_argument("--floormat", default=None,
                    help="image laid as a court-centre mat decal on the floor")
    ap.add_argument("--floormat-width", type=float, default=6.0,
                    help="mat width in metres; height follows the image aspect")
    ap.add_argument("--floormat-at", default="0,0",
                    help="semicolon-separated x,y centres, one mat each, "
                         "e.g. '-12,0;0,0;12,0' for every court of the strip")
    return ap.parse_args(script_argv())


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_stl(path):
    """Blender renamed the STL operator in 4.x; support both."""
    before = set(bpy.data.objects)
    if hasattr(bpy.ops.wm, "stl_import"):
        bpy.ops.wm.stl_import(filepath=path)
    else:
        bpy.ops.import_mesh.stl(filepath=path)
    new = list(set(bpy.data.objects) - before)
    if not new:
        raise RuntimeError(f"STL import produced nothing: {path}")
    obj = new[0]
    mesh = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)   # keep the data, drop the object
    return mesh


def mat_metal(name, base, rough, metallic=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*base, 1.0)
    b.inputs["Metallic"].default_value = metallic
    b.inputs["Roughness"].default_value = rough
    return m


def mat_floor():
    """White, polished, slightly rough: a real specular floor with true reflections.

    This is the single biggest difference from the MuJoCo version, where the floor
    'reflection' was a flat planar fake.
    """
    m = bpy.data.materials.new("arena_floor")
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.90, 0.91, 0.93, 1.0)
    b.inputs["Metallic"].default_value = 0.0
    b.inputs["Roughness"].default_value = 0.13
    if "Specular IOR Level" in b.inputs:
        b.inputs["Specular IOR Level"].default_value = 0.62
    elif "Specular" in b.inputs:
        b.inputs["Specular"].default_value = 0.62
    return m


def mat_banner(name, image_path):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(image_path)
    tex.interpolation = "Cubic"
    nt.links.new(tex.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.45
    b.inputs["Metallic"].default_value = 0.0
    return m


def build_lighting(scale=1.0):
    """Large soft key + cool rim + overhead wash, plus a bright world.

    Area lights, not point lights: soft shadows are most of why a Cycles frame reads
    as photographed rather than rendered.
    """
    world = bpy.data.worlds.new("W")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.55, 0.60, 0.68, 1.0)
    bg.inputs["Strength"].default_value = 0.55 * scale

    def area(name, loc, rot, size, energy, color):
        ld = bpy.data.lights.new(name, type="AREA")
        ld.size = size
        ld.energy = energy * scale
        ld.color = color
        ob = bpy.data.objects.new(name, ld)
        ob.location = loc
        ob.rotation_euler = rot
        bpy.context.collection.objects.link(ob)
        return ob

    # Blender area-light power is in Watts and falls off with distance squared. At
    # ~20 m these lights need tens of thousands, not hundreds of thousands: the first
    # pass used 220 kW and returned a white frame.
    area("key", (-14, -16, 16), (np.radians(48), 0, np.radians(-42)), 16.0,
         26000, (1.0, 0.97, 0.93))
    area("rim", (16, 20, 12), (np.radians(-58), 0, np.radians(140)), 14.0,
         14000, (0.80, 0.87, 1.0))
    area("top", (0, 0, 22), (0, 0, 0), 30.0, 11000, (1.0, 1.0, 1.0))


def main():
    a = parse()
    clear_scene()
    scene_path = os.path.join(a.export, "scene.json")
    S = json.load(open(scene_path))
    anim = np.load(os.path.join(a.export, "anim.npz"))
    cam = np.load(os.path.join(a.export, "camera.npz"))
    P, Q = anim["pos"], anim["quat"]
    eye, target, fovy = cam["eye"], cam["target"], cam["fovy"]
    nF = P.shape[0]
    print(f"[arena] {P.shape[1]} instances, {nF} frames")

    # ---- shared mesh data -------------------------------------------------
    # Build meshes from the arrays MuJoCo compiled. Importing the source STLs put
    # every part at its own centre-of-mass offset and the robots came out exploded.
    MA = np.load(os.path.join(a.export, "meshes.npz"))
    mesh_cache = {}
    for name in S["meshes"]:
        v = MA[f"v_{name}"]
        f = MA[f"f_{name}"]
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(x) for x in v.tolist()], [],
                       [tuple(x) for x in f.tolist()])
        me.validate()
        me.update()
        me.shade_smooth()
        mesh_cache[name] = me
    print(f"[arena] built {len(mesh_cache)} meshes from compiled geometry "
          f"({sum(len(m.vertices) for m in mesh_cache.values())} verts)")

    shell = mat_metal("g1_shell", (0.62, 0.64, 0.68), 0.28, 1.0)
    dark = mat_metal("g1_dark", (0.05, 0.055, 0.065), 0.45, 0.85)

    objs = []
    for mg in S["mesh_geoms"]:
        me = mesh_cache[mg["mesh"]]
        ob = bpy.data.objects.new(f"g{mg['geom']}", me)
        lum = sum(mg["rgba"][:3]) / 3.0
        ob.data.materials.clear()
        ob.data.materials.append(shell if lum > 0.35 else dark)
        ob.rotation_mode = "QUATERNION"
        bpy.context.collection.objects.link(ob)
        objs.append(ob)
    # material assignment lives on shared mesh data, so set it once per mesh instead
    for name, me in mesh_cache.items():
        if not me.materials:
            me.materials.append(shell)

    # ---- static set: floor, zone lines, barriers, banners -----------------
    floor_mat, line_mat = mat_floor(), mat_metal("zone_line", (0.10, 0.12, 0.17), 0.5, 0.0)
    barrier_mat = mat_metal("barrier", (0.86, 0.87, 0.90), 0.35, 0.0)
    ban_mats = {k: mat_banner(k, os.path.join(a.assets, os.path.basename(os.path.dirname(v)),
                                              os.path.basename(v)))
                for k, v in S["banners"].items()}

    bpy.ops.mesh.primitive_plane_add(size=140, location=(0, 0, 0))
    fl = bpy.context.active_object
    fl.data.materials.append(floor_mat)

    if a.floormat:
        # Court-centre mat: a decal plane a few millimetres above the floor so it
        # never z-fights, matte (velvet) so the glossy floor reflections don't
        # smear the logo. Plane UVs already span the image; scaling the object to
        # the image aspect keeps every texel square.
        img = bpy.data.images.load(a.floormat)
        mw = a.floormat_width
        mh = mw * img.size[1] / img.size[0]
        m = bpy.data.materials.new("floor_mat_decal")
        m.use_nodes = True
        nt = m.node_tree
        b = nt.nodes["Principled BSDF"]
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img
        tex.interpolation = "Cubic"
        tex.extension = "EXTEND"
        nt.links.new(tex.outputs["Color"], b.inputs["Base Color"])
        b.inputs["Roughness"].default_value = 0.85
        b.inputs["Metallic"].default_value = 0.0
        centres = [tuple(float(c) for c in p.split(","))
                   for p in a.floormat_at.split(";") if p.strip()]
        for cx, cy in centres:
            bpy.ops.mesh.primitive_plane_add(size=1, location=(cx, cy, 0.006))
            mat_ob = bpy.context.active_object
            mat_ob.scale = (mw / 2.0, mh / 2.0, 1.0)
            mat_ob.data.materials.append(m)
        print(f"[arena] {len(centres)} floor mat(s) {mw:.2f}x{mh:.2f} m "
              f"at {centres} from {a.floormat}")

    for st in S["statics"]:
        if st["name"] == "floor":
            continue
        kind, sz = st["kind"], st["size"]
        if kind == 6:                                   # box
            bpy.ops.mesh.primitive_cube_add(size=2, location=st["pos"])
            ob = bpy.context.active_object
            ob.scale = (max(sz[0], 1e-4), max(sz[1], 1e-4), max(sz[2], 1e-4))
            mat = barrier_mat if st["material"] == "barrier" else line_mat
            ob.data.materials.append(mat)
        elif kind == 0:                                 # plane -> banner board
            bpy.ops.mesh.primitive_plane_add(size=2, location=st["pos"])
            ob = bpy.context.active_object
            ob.scale = (max(sz[0], 1e-4), max(sz[1], 1e-4), 1.0)
            ob.data.materials.append(ban_mats.get(st["material"], barrier_mat))
        else:
            continue
        ob.rotation_mode = "QUATERNION"
        ob.rotation_quaternion = mathutils.Quaternion(st["quat"])

    build_lighting(a.lightscale)

    # ---- camera -----------------------------------------------------------
    cd = bpy.data.cameras.new("cam")
    cd.sensor_fit = "VERTICAL"
    cam_ob = bpy.data.objects.new("cam", cd)
    bpy.context.collection.objects.link(cam_ob)
    bpy.context.scene.camera = cam_ob
    cd.dof.use_dof = not a.nodof
    cd.dof.aperture_fstop = 2.8

    def set_frame(scene_, _depsgraph=None):
        i = int(np.clip(scene_.frame_current - 1, 0, nF - 1))
        for k, ob in enumerate(objs):
            ob.location = P[i, k]
            ob.rotation_quaternion = Q[i, k]
        e, t = mathutils.Vector(eye[i]), mathutils.Vector(target[i])
        cam_ob.location = e
        d = (t - e)
        cam_ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        cd.angle_y = np.radians(float(fovy[i]))
        cd.dof.focus_distance = max(0.5, d.length)

    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(set_frame)

    # ---- render settings --------------------------------------------------
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = a.samples
    sc.cycles.use_denoising = True
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.01
    sc.cycles.max_bounces = a.bounces
    sc.cycles.transmission_bounces = min(4, a.bounces)
    sc.cycles.adaptive_min_samples = max(8, a.samples // 4)
    sc.render.resolution_x = a.width
    sc.render.resolution_y = a.height
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.compression = 15
    # AgX holds specular highlights on white metal instead of clipping them flat.
    # Ask for it by assigning, not by probing: this enum is filled in from the OCIO
    # config at runtime and does not introspect -- bl_rna reports only ['NONE'] even
    # though "AgX" assigns fine. The old probe therefore always fell through to
    # Standard, a hard clamp, and that is what returned near-white frames (91% of
    # pixels pinned at 255) regardless of how far the lights were turned down.
    for vt in ("AgX", "Filmic", "Standard"):
        try:
            sc.view_settings.view_transform = vt
        except TypeError:
            continue
        break
    sc.view_settings.look = "None"
    sc.view_settings.exposure = a.exposure
    print(f"[arena] view transform {sc.view_settings.view_transform}, "
          f"exposure {a.exposure:+.2f}, lightscale {a.lightscale:g}")
    sc.frame_start = a.start + 1
    sc.frame_end = (nF if a.end < 0 else min(a.end, nF))
    os.makedirs(a.out, exist_ok=True)
    sc.render.filepath = os.path.join(a.out, "f_")

    if a.device != "CPU":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        n = 0
        # A build without this backend rejects the enum outright. Falling back to
        # CPU beats dying at the last step of a long setup.
        try:
            prefs.compute_device_type = a.device
        except TypeError:
            print(f"[arena] {a.device} unsupported by this build, using CPU")
        else:
            prefs.get_devices()
            for dev in prefs.devices:
                dev.use = (dev.type == a.device)
                n += dev.use
        sc.cycles.device = "GPU" if n else "CPU"
        print(f"[arena] {a.device}: {n} device(s), cycles.device={sc.cycles.device}")

    print(f"[arena] rendering frames {sc.frame_start}..{sc.frame_end} "
          f"at {a.width}x{a.height}, {a.samples} samples")
    bpy.ops.render.render(animation=True)
    print("[arena] done")


main()
