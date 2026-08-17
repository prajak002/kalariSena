"""MuJoCo runtime for the Unitree G1 29-DoF: PD torque control, contacts, video.

Physics runs at 500 Hz (dt = 0.002, the official MJCF's own timestep) with 50 Hz
control decimation, i.e. 10 physics substeps per control tick.

Control is explicit PD torque:  tau = kp * (q_cmd - q) - kd * dq
applied through the MJCF's direct-drive torque motors (gear=1, no internal
position servo), then clipped to the per-joint effort limits declared in the
official URDF. The MJCF itself leaves ctrlrange/forcerange unset, so the limits
are read from g1_29dof_rev_1_0.urdf at load time rather than invented here.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

# Offscreen GL backend. EGL and OSMesa are Linux-only; on macOS MuJoCo's CGL
# backend renders offscreen natively. Try in order and keep the first that works.
_GL_CANDIDATES = ("egl", "osmesa", "glfw")


def select_gl_backend(verbose: bool = True) -> str | None:
    """Pick a working offscreen GL backend, or None if rendering is impossible."""
    if os.environ.get("MUJOCO_GL"):
        return os.environ["MUJOCO_GL"]
    import platform

    if platform.system() == "Darwin":
        # macOS: default (CGL) is the only offscreen path; egl/osmesa hard-error.
        try:
            _probe_render()
            if verbose:
                print("[gl] using macOS default CGL backend (egl/osmesa are Linux-only)")
            return "cgl-default"
        except Exception as exc:  # pragma: no cover
            if verbose:
                print(f"[gl] CGL probe failed: {exc}")
            return None
    for backend in _GL_CANDIDATES:
        os.environ["MUJOCO_GL"] = backend
        try:
            _probe_render()
            if verbose:
                print(f"[gl] using MUJOCO_GL={backend}")
            return backend
        except Exception as exc:
            if verbose:
                print(f"[gl] backend {backend} failed: {exc}")
            os.environ.pop("MUJOCO_GL", None)
    return None


def _probe_render() -> None:
    import mujoco

    m = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><light pos="0 0 3"/>'
        '<geom type="box" size=".1 .1 .1"/></worldbody></mujoco>'
    )
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 64, 64)
    r.update_scene(d)
    r.render()
    r.close()


# Per-joint effort limits, parsed from the official URDF (see _load_torque_limits).
DEFAULT_URDF = "assets/unitree_g1/g1_29dof_rev_1_0.urdf"
DEFAULT_SCENE = "assets/unitree_g1/scene_29dof.xml"

# G1 nominal standing posture (slight knee/ankle flexion), joint-space authored.
DEFAULT_STANDING_POSE: dict[str, float] = {
    "left_hip_pitch_joint": -0.20,
    "left_knee_joint": 0.42,
    "left_ankle_pitch_joint": -0.23,
    "right_hip_pitch_joint": -0.20,
    "right_knee_joint": 0.42,
    "right_ankle_pitch_joint": -0.23,
    "left_shoulder_pitch_joint": 0.25,
    "left_shoulder_roll_joint": 0.18,
    "left_elbow_joint": 0.90,
    "right_shoulder_pitch_joint": 0.25,
    "right_shoulder_roll_joint": -0.18,
    "right_elbow_joint": 0.90,
}


@dataclass
class PDGains:
    """Joint-group PD gains.

    Two modes:

    * Flat group gains (kp_leg/kp_waist/kp_arm, kd_*). This is the literal
      starting point from the brief, kept for reference and comparison.

    * `inertia_scaled=True` (default): gains are derived per joint from the
      actual mass-matrix diagonal at the standing pose,
          kp_i = I_i * omega^2,     kd_i = 2 * zeta * I_i * omega
      which gives every joint the same closed-loop natural frequency and damping
      ratio regardless of how heavy it is.

      This is not cosmetic tuning. The official G1 MJCF declares no joint
      armature, so the wrist joints have effective inertia I ~ 1e-3 kg m^2. An
      explicitly-integrated damping term is only stable while kd * dt / I < 2,
      i.e. kd < 1.0 for those joints at dt = 0.002 s. A flat kd = 2.5 sits a
      factor of 5 past that limit, so the wrists diverge within ~0.1 s and pump
      enough energy into the floating base to throw the whole robot over. Flat
      gains therefore fail at *every* kp; see results_sim/gain_study.csv.

      kd is additionally hard-clamped to `stability_fraction * I / dt` so this
      failure mode cannot reappear if omega/zeta are retuned.
    """

    # Defaults are the lowest-effort configuration that actually stands, chosen
    # by scripts/gain_study.py (see results_sim/gain_study.csv). The brief's
    # starting point (80/40/30, kd 2.5) is in that CSV as a failing row.
    kp_leg: float = 300.0
    kp_waist: float = 150.0
    kp_arm: float = 60.0
    kd_leg: float = 8.0
    kd_waist: float = 8.0
    kd_arm: float = 8.0

    inertia_scaled: bool = False
    omega: float = 26.0            # rad/s closed-loop natural frequency
    zeta: float = 1.1              # damping ratio (slightly overdamped)
    stability_fraction: float = 1.0  # kd <= stability_fraction * I / dt

    def kp_for(self, joint_name: str) -> float:
        if any(k in joint_name for k in ("hip", "knee", "ankle")):
            return self.kp_leg
        if "waist" in joint_name:
            return self.kp_waist
        return self.kp_arm

    def kd_for(self, joint_name: str) -> float:
        if any(k in joint_name for k in ("hip", "knee", "ankle")):
            return self.kd_leg
        if "waist" in joint_name:
            return self.kd_waist
        return self.kd_arm


@dataclass
class ContactReport:
    left_contact: bool
    right_contact: bool
    left_force: float
    right_force: float
    total_normal_force: float
    left_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    right_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))


def _load_torque_limits(urdf_path: str) -> dict[str, float]:
    root = ET.parse(urdf_path).getroot()
    limits: dict[str, float] = {}
    for joint in root.iter("joint"):
        lim = joint.find("limit")
        if lim is not None and lim.get("effort") is not None:
            limits[joint.get("name")] = float(lim.get("effort"))
    if not limits:
        raise ValueError(f"no joint effort limits found in {urdf_path}")
    return limits


class G1MujocoRuntime:
    """Loads the official G1 MJCF and exposes PD torque control + contact reads."""

    CONTROL_HZ = 50.0

    # Rotor/gearbox inertia reflected to the joint. The 29-DoF MJCF omits it,
    # but Unitree's own g1_12dof.xml (the model their RL stack ships with) sets
    # armature="0.01" on every leg joint, so this is the vendor's own value
    # rather than a number invented here. Without it the ankle and wrist joints
    # have effective inertia ~4e-4 kg m^2, which is both physically wrong for a
    # harmonic-drive joint and numerically unstable under explicit PD damping.
    DEFAULT_ARMATURE = 0.01

    def __init__(
        self,
        scene_path: str = DEFAULT_SCENE,
        urdf_path: str = DEFAULT_URDF,
        gains: PDGains | None = None,
        armature: float | None = DEFAULT_ARMATURE,
    ):
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.gains = gains or PDGains()
        self.armature = armature

        self.dt = float(self.model.opt.timestep)
        self.decimation = int(round((1.0 / self.CONTROL_HZ) / self.dt))
        if self.decimation < 1:
            raise ValueError(f"physics dt {self.dt} too coarse for {self.CONTROL_HZ} Hz control")

        # --- actuator name -> index, and the joint each actuator drives -------
        self.actuator_names: list[str] = []
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name is None:
                raise ValueError(f"actuator {i} is unnamed; cannot build a name-keyed map")
            self.actuator_names.append(name)
        self.act_index = {n: i for i, n in enumerate(self.actuator_names)}

        # Joint qpos/qvel address for each actuator, resolved through the
        # actuator's transmission target (never assumed to be index order).
        self.act_qadr = np.zeros(self.model.nu, dtype=int)
        self.act_vadr = np.zeros(self.model.nu, dtype=int)
        self.act_joint_names: list[str] = []
        for i in range(self.model.nu):
            jid = int(self.model.actuator_trnid[i, 0])
            jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            self.act_joint_names.append(jname)
            self.act_qadr[i] = int(self.model.jnt_qposadr[jid])
            self.act_vadr[i] = int(self.model.jnt_dofadr[jid])

        # --- armature (reflected rotor inertia) ------------------------------
        if self.armature is not None:
            for i in range(self.model.nu):
                self.model.dof_armature[self.act_vadr[i]] = self.armature

        # --- torque limits ---------------------------------------------------
        # Preferred source is the MJCF itself, which declares per-joint
        # actuatorfrcrange. The URDF effort limits are parsed as a cross-check;
        # a mismatch means the two asset files have diverged.
        urdf_limits = _load_torque_limits(urdf_path)
        missing = [n for n in self.act_joint_names if n not in urdf_limits]
        if missing:
            raise ValueError(f"no URDF effort limit for joints: {missing}")
        urdf_tau = np.array([urdf_limits[n] for n in self.act_joint_names], dtype=np.float64)

        mjcf_tau = np.zeros(self.model.nu)
        for i in range(self.model.nu):
            jid = int(self.model.actuator_trnid[i, 0])
            if bool(self.model.jnt_actfrclimited[jid]):
                mjcf_tau[i] = float(self.model.jnt_actfrcrange[jid, 1])
        if np.all(mjcf_tau > 0):
            self.torque_limit = mjcf_tau
            self.torque_limit_source = "MJCF jnt_actfrcrange"
            disagree = np.abs(mjcf_tau - urdf_tau) > 1e-6
            if disagree.any():
                bad = [self.act_joint_names[i] for i in np.flatnonzero(disagree)]
                raise ValueError(f"MJCF and URDF torque limits disagree for: {bad}")
        else:
            self.torque_limit = urdf_tau
            self.torque_limit_source = "URDF effort"

        self.kp = np.array([self.gains.kp_for(n) for n in self.act_joint_names])
        self.kd = np.array([self.gains.kd_for(n) for n in self.act_joint_names])
        self.joint_inertia = np.zeros(self.model.nu)  # filled by _apply_inertia_scaling

        # --- foot contact geoms: real MJCF collision spheres, by body ---------
        self.left_foot_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link"
        )
        self.right_foot_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link"
        )
        if self.left_foot_body < 0 or self.right_foot_body < 0:
            raise ValueError("ankle_roll_link bodies not found in model")
        self.left_foot_geoms = self._collision_geoms_of_body(self.left_foot_body)
        self.right_foot_geoms = self._collision_geoms_of_body(self.right_foot_body)
        if not self.left_foot_geoms or not self.right_foot_geoms:
            raise ValueError("no collision geoms found on the ankle_roll_link bodies")
        self.floor_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        # Body-frame offsets of the foot contact spheres, read from the MJCF.
        # These define the true support polygon (a ~17cm x 6cm foot), replacing
        # any fixed-size placeholder square.
        self.left_foot_offsets = np.array(
            [self.model.geom_pos[g] for g in self.left_foot_geoms], dtype=np.float64
        )
        self.right_foot_offsets = np.array(
            [self.model.geom_pos[g] for g in self.right_foot_geoms], dtype=np.float64
        )

        # --- impact bodies for the fall A/B experiment ------------------------
        self.head_geoms = self._geoms_with_mesh("head_link")
        self.torso_geoms = self._geoms_with_mesh("torso_link") + self._geoms_with_mesh("logo_link")
        self.pelvis_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

        self.total_mass = float(self.model.body_mass.sum())
        self._renderer = None

        # Always measure the mass-matrix diagonal (needed for the stability
        # diagnostic kd*dt/I); only override kp/kd when inertia scaling is on.
        self._apply_inertia_scaling(override_gains=self.gains.inertia_scaled)

    def _apply_inertia_scaling(self, override_gains: bool = True) -> None:
        """Measure the mass-matrix diagonal; optionally set kp/kd from it."""
        mujoco = self.mujoco
        saved_q = self.data.qpos.copy()
        saved_v = self.data.qvel.copy()
        self.data.qpos[:] = self.default_qpos()
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Mass-matrix diagonal via M @ e_v, which avoids depending on whether
        # this MuJoCo build stores M densely (data.M) or sparsely (data.qM).
        inertia = np.zeros(self.model.nu, dtype=np.float64)
        probe = np.zeros(self.model.nv)
        result = np.zeros(self.model.nv)
        for i, v in enumerate(self.act_vadr):
            probe[:] = 0.0
            probe[v] = 1.0
            mujoco.mj_mulM(self.model, self.data, result, probe)
            inertia[i] = float(result[v])
        if not np.all(inertia > 0):
            raise ValueError("non-positive mass-matrix diagonal; cannot scale gains")

        self.data.qpos[:] = saved_q
        self.data.qvel[:] = saved_v
        mujoco.mj_forward(self.model, self.data)

        self.joint_inertia = inertia
        if not override_gains:
            return
        g = self.gains
        self.kp = inertia * g.omega**2
        kd = 2.0 * g.zeta * inertia * g.omega
        # Explicit-integration stability bound: kd * dt / I must stay below ~2.
        kd_max = g.stability_fraction * inertia / self.dt
        self.kd_unclamped = kd.copy()
        self.kd = np.minimum(kd, kd_max)

    def gain_table(self) -> str:
        rows = [f"{'joint':28s} {'I (kg m^2)':>11s} {'kp':>9s} {'kd':>7s} "
                f"{'kd*dt/I':>8s} {'tau_lim':>8s}"]
        for i, n in enumerate(self.act_joint_names):
            inertia = self.joint_inertia[i] if self.joint_inertia[i] else float("nan")
            rows.append(
                f"{n:28s} {inertia:11.5f} {self.kp[i]:9.2f} {self.kd[i]:7.3f} "
                f"{self.kd[i] * self.dt / inertia:8.3f} {self.torque_limit[i]:8.1f}"
            )
        return "\n".join(rows)

    # ------------------------------------------------------------- helpers ---
    def _collision_geoms_of_body(self, body_id: int) -> list[int]:
        out = []
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] != body_id:
                continue
            if self.model.geom_contype[g] == 0 and self.model.geom_conaffinity[g] == 0:
                continue
            out.append(g)
        return out

    def _geoms_with_mesh(self, mesh_name: str) -> list[int]:
        mujoco = self.mujoco
        mesh_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MESH, mesh_name)
        if mesh_id < 0:
            return []
        return [
            g
            for g in range(self.model.ngeom)
            if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
            and self.model.geom_dataid[g] == mesh_id
            and not (self.model.geom_contype[g] == 0 and self.model.geom_conaffinity[g] == 0)
        ]

    # --------------------------------------------------------------- state ---
    def default_qpos(self, base_height: float | None = None,
                     clearance: float = 0.001) -> np.ndarray:
        """Standing pose. Base height is *solved* so the feet rest on the floor.

        Hard-coding a base height for a given joint configuration either buries
        the feet (instant huge contact forces) or drops the robot (impact
        transient). Instead we place the pose, run forward kinematics, and shift
        the base so the lowest foot contact sphere sits `clearance` above z = 0.
        """
        qpos = np.zeros(self.model.nq)
        qpos[2] = 1.0
        qpos[3] = 1.0  # quaternion [w, x, y, z] = identity
        for name, angle in DEFAULT_STANDING_POSE.items():
            if name not in self.act_index:
                raise KeyError(f"standing pose references unknown joint {name}")
            qpos[self.act_qadr[self.act_index[name]]] = angle
        if base_height is not None:
            qpos[2] = base_height
            return qpos

        saved_q = self.data.qpos.copy()
        saved_v = self.data.qvel.copy()
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        foot_geoms = self.left_foot_geoms + self.right_foot_geoms
        lowest = min(
            float(self.data.geom_xpos[g][2]) - float(self.model.geom_size[g][0])
            for g in foot_geoms
        )
        self.data.qpos[:] = saved_q
        self.data.qvel[:] = saved_v
        self.mujoco.mj_forward(self.model, self.data)

        qpos[2] += clearance - lowest
        return qpos

    def default_joint_targets(self) -> np.ndarray:
        """Standing pose expressed in actuator order."""
        return self.default_qpos()[self.act_qadr]

    def reset(self, qpos: np.ndarray | None = None, qvel: np.ndarray | None = None) -> None:
        self.mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.default_qpos() if qpos is None else qpos
        self.data.qvel[:] = 0.0 if qvel is None else qvel
        self.data.qacc[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

    def settle(self, seconds: float, q_cmd: np.ndarray | None = None) -> None:
        """Run PD control for `seconds` to let the robot settle onto the floor."""
        target = self.default_joint_targets() if q_cmd is None else q_cmd
        for _ in range(int(round(seconds * self.CONTROL_HZ))):
            self.control_step(target)

    # ------------------------------------------------------------- control ---
    def pd_torque(self, q_cmd: np.ndarray) -> np.ndarray:
        q = self.data.qpos[self.act_qadr]
        dq = self.data.qvel[self.act_vadr]
        tau = self.kp * (np.asarray(q_cmd, dtype=np.float64) - q) - self.kd * dq
        return np.clip(tau, -self.torque_limit, self.torque_limit)

    def control_step(self, q_cmd: np.ndarray, xfrc: np.ndarray | None = None) -> np.ndarray:
        """One 50 Hz control tick = `decimation` physics substeps at 500 Hz.

        Torque is recomputed every physics substep (the PD loop itself runs at
        500 Hz); only the *target* is held for the duration of the tick.
        """
        if xfrc is not None:
            self.data.xfrc_applied[:] = xfrc
        tau_last = np.zeros(self.model.nu)
        for _ in range(self.decimation):
            tau_last = self.pd_torque(q_cmd)
            self.data.ctrl[:] = tau_last
            self.mujoco.mj_step(self.model, self.data)
        return tau_last

    # ------------------------------------------------------------ contacts ---
    def foot_contacts(self, force_threshold: float = 1.0) -> ContactReport:
        """Foot contact flags from real MuJoCo contacts on the ankle_roll geoms.

        Not a height heuristic: a contact counts only if MuJoCo actually created
        it between a foot collision geom and the floor, and the normal force
        exceeds `force_threshold` newtons.
        """
        mujoco = self.mujoco
        left_f = right_f = total_f = 0.0
        left_pts: list[np.ndarray] = []
        right_pts: list[np.ndarray] = []
        wrench = np.zeros(6)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            mujoco.mj_contactForce(self.model, self.data, i, wrench)
            fn = float(abs(wrench[0]))  # normal component, contact frame
            total_f += fn
            g1, g2 = int(con.geom1), int(con.geom2)
            if g1 in self.left_foot_geoms or g2 in self.left_foot_geoms:
                left_f += fn
                if fn > force_threshold:
                    left_pts.append(np.array(con.pos, dtype=np.float64))
            if g1 in self.right_foot_geoms or g2 in self.right_foot_geoms:
                right_f += fn
                if fn > force_threshold:
                    right_pts.append(np.array(con.pos, dtype=np.float64))
        return ContactReport(
            left_contact=left_f > force_threshold,
            right_contact=right_f > force_threshold,
            left_force=left_f,
            right_force=right_f,
            total_normal_force=total_f,
            left_points=np.array(left_pts) if left_pts else np.zeros((0, 3)),
            right_points=np.array(right_pts) if right_pts else np.zeros((0, 3)),
        )

    def geom_group_force(self, geom_ids: list[int]) -> float:
        """Summed contact-force magnitude on a set of geoms (newtons)."""
        if not geom_ids:
            return 0.0
        mujoco = self.mujoco
        wrench = np.zeros(6)
        total = 0.0
        gset = set(geom_ids)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if int(con.geom1) in gset or int(con.geom2) in gset:
                mujoco.mj_contactForce(self.model, self.data, i, wrench)
                total += float(np.linalg.norm(wrench[:3]))
        return total

    def foot_support_points(self, contacts: ContactReport) -> np.ndarray:
        """World-frame XY of every contact sphere on each *contacting* foot.

        Uses the full sphere set of a foot in contact (not only the spheres that
        happen to be touching this instant), which is the standard support-polygon
        definition for a rigid foot resting on the ground.
        """
        mujoco = self.mujoco
        pts: list[np.ndarray] = []
        for in_contact, geoms in (
            (contacts.left_contact, self.left_foot_geoms),
            (contacts.right_contact, self.right_foot_geoms),
        ):
            if not in_contact:
                continue
            for g in geoms:
                pts.append(np.array(self.data.geom_xpos[g], dtype=np.float64))
        return np.array(pts) if pts else np.zeros((0, 3))

    # ------------------------------------------------------------ readouts ---
    @property
    def base_height(self) -> float:
        return float(self.data.qpos[2])

    @property
    def base_quat_wxyz(self) -> np.ndarray:
        return np.array(self.data.qpos[3:7], dtype=np.float64)

    def torso_upright_cos(self) -> float:
        """cos(angle) between the pelvis z-axis and world up. 1.0 = upright."""
        rot = self.data.xmat[self.pelvis_body].reshape(3, 3)
        return float(rot[2, 2])

    def mj_subtree_com(self) -> np.ndarray:
        """Whole-robot CoM from MuJoCo, for cross-checking against Pinocchio."""
        return np.array(self.data.subtree_com[self.pelvis_body], dtype=np.float64)

    # ------------------------------------------------------------ rendering ---
    def make_renderer(self, width: int = 640, height: int = 480):
        import mujoco

        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height, width)
        return self._renderer

    def render_frame(self, camera: str = "track", width: int = 640, height: int = 480):
        r = self.make_renderer(width, height)
        r.update_scene(self.data, camera=camera)
        return r.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
