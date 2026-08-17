"""The closed-loop controller: state -> Pinocchio features -> ModeSwitch -> PD targets.

This is the single implementation of the control tick used by every experiment
(tracking, push sweep, fall A/B), so all of them report identical quantities.

Per control tick:
  1. Read MuJoCo state and real foot contacts (contact-based, not a height test).
  2. Convert to Pinocchio conventions and compute CoM, centroidal momentum,
     support polygon and capture-point margin from the real foot geometry.
  3. Feed those features to ModeSwitch.
  4. Emit PD targets: the reference motion in NOMINAL, or a protective crouch in
     FALL/RECOVERY.

The protective crouch is a SCRIPTED, hand-designed posture. It is not learned
and must never be described as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.dynamics.pinocchio_wrapper import PinocchioWrapper
from src.sim.conventions import StateConverter
from src.sim.motions import Motion
from src.sim.mujoco_runtime import G1MujocoRuntime
from src.switch.mode_switch import Mode, ModeSwitch, SwitchConfig

# Scripted protective response: drop the CoM, tuck the elbows in front of the
# head/chest, and round the trunk forward. Hand-designed, not optimised.
PROTECTIVE_CROUCH: dict[str, float] = {
    "left_hip_pitch_joint": -1.30, "right_hip_pitch_joint": -1.30,
    "left_knee_joint": 2.10, "right_knee_joint": 2.10,
    "left_ankle_pitch_joint": -0.75, "right_ankle_pitch_joint": -0.75,
    "left_hip_roll_joint": 0.12, "right_hip_roll_joint": -0.12,
    "waist_pitch_joint": 0.42, "waist_yaw_joint": 0.0, "waist_roll_joint": 0.0,
    # Arms tucked: shoulders forward and inward, elbows deeply flexed so the
    # forearms shield the head and the elbows take the impact instead.
    "left_shoulder_pitch_joint": -1.55, "right_shoulder_pitch_joint": -1.55,
    "left_shoulder_roll_joint": 0.55, "right_shoulder_roll_joint": -0.55,
    "left_shoulder_yaw_joint": -0.35, "right_shoulder_yaw_joint": 0.35,
    "left_elbow_joint": 2.20, "right_elbow_joint": 2.20,
    "left_wrist_pitch_joint": 0.0, "right_wrist_pitch_joint": 0.0,
}

# Upright/fallen thresholds used to build the ModeSwitch feature dict.
UPRIGHT_COS = 0.80
FALLEN_COS = 0.35
FALLEN_HEIGHT = 0.35


@dataclass
class BalanceGains:
    """CoM-feedback balance controller (ankle + hip strategy).

    Physics-first and model-free of any learning: the reference joint targets are
    corrected by a term proportional to where the CoM (and its capture point) sits
    relative to the centre of the measured support polygon.

        e     = com_xy - support_centre_xy      (position error)
        edot  = com_velocity_xy

        ankle_pitch += -(k_ankle * e_x + kd_ankle * edot_x)   fore/aft
        ankle_roll  += +(k_ankle * e_y + kd_ankle * edot_y)   lateral
        hip_roll    += -(k_hip   * e_y + kd_hip   * edot_y)   lateral pelvis shift

    Disabled by default so that every result produced before it existed is
    reproducible unchanged. Enable with KalariController(..., balance=BalanceGains()).
    """

    enabled: bool = True
    k_ankle: float = 1.6
    kd_ankle: float = 0.45
    k_hip: float = 1.2
    kd_hip: float = 0.30
    max_ankle_offset: float = 0.35   # rad, keeps the correction inside joint range
    max_hip_offset: float = 0.30     # rad


@dataclass
class StepLog:
    """One control tick. Schema is shared by every results CSV."""

    step: int = 0
    t: float = 0.0
    motion_id: str = ""
    phase: float = 0.0
    phase_label: str = ""
    mode: str = "nominal"
    # base
    base_x: float = 0.0
    base_y: float = 0.0
    base_z: float = 0.0
    upright_cos: float = 0.0
    # CoM / momentum
    com_x: float = 0.0
    com_y: float = 0.0
    com_z: float = 0.0
    com_vx: float = 0.0
    com_vy: float = 0.0
    com_vz: float = 0.0
    lin_momentum_norm: float = 0.0
    ang_momentum_norm: float = 0.0
    momentum_norm: float = 0.0
    # support / stability
    cp_x: float = 0.0
    cp_y: float = 0.0
    cp_margin: float = 0.0
    com_margin: float = 0.0
    support_area: float = 0.0
    support_mode: str = ""
    contact_left: int = 0
    contact_right: int = 0
    # sim extras
    grf_left: float = 0.0
    grf_right: float = 0.0
    grf_total: float = 0.0
    torque_norm: float = 0.0
    torque_max: float = 0.0
    tracking_err_rms: float = 0.0
    head_force: float = 0.0
    torso_force: float = 0.0
    push_force: float = 0.0


def _feature_dict(log: StepLog, step: int) -> dict:
    return {
        "cp_margin": log.cp_margin,
        "momentum_norm": log.momentum_norm,
        "base_height": log.base_z,
        "is_fallen": log.upright_cos < FALLEN_COS or log.base_z < FALLEN_HEIGHT,
        "is_upright": log.upright_cos > UPRIGHT_COS and log.base_z > 0.55,
        "step_index": step,
    }


class KalariController:
    """Physics-first controller: Pinocchio features + ModeSwitch + PD tracking."""

    def __init__(
        self,
        runtime: G1MujocoRuntime,
        pin_wrapper: PinocchioWrapper,
        switch_cfg: SwitchConfig | None = None,
        protective_response: bool = True,
        balance: BalanceGains | None = None,
    ):
        self.rt = runtime
        self.pin = pin_wrapper
        self.conv = StateConverter(runtime.model, pin_wrapper.model)
        self.switch = ModeSwitch(switch_cfg or SwitchConfig())
        self.protective_response = protective_response
        self.balance = balance
        self._support_centre = np.zeros(2)

        self.crouch_targets = runtime.default_joint_targets().copy()
        for name, angle in PROTECTIVE_CROUCH.items():
            if name in self.rt.act_index:
                self.crouch_targets[self.rt.act_index[name]] = angle

        # Reference joint order (motion) -> actuator order.
        self._motion_to_act: np.ndarray | None = None

    def bind_motion(self, motion: Motion) -> None:
        self._motion_to_act = np.array(
            [motion.joint_names.index(n) for n in self.rt.act_joint_names], dtype=int
        )

    def measure(self, step: int, t: float, motion: Motion | None = None,
                push_force: float = 0.0) -> StepLog:
        """Compute every logged feature from the current simulator state."""
        rt = self.rt
        qpos = np.array(rt.data.qpos, dtype=np.float64)
        qvel = np.array(rt.data.qvel, dtype=np.float64)
        q_pin = self.conv.mj_to_pin_q(qpos)
        v_pin = self.conv.mj_to_pin_v(qpos, qvel)

        contacts = rt.foot_contacts()
        com, com_vel = self.pin.compute_com(q_pin, v_pin)
        hg = self.pin.compute_centroidal_momentum(q_pin, v_pin)
        support_pts = rt.foot_support_points(contacts)
        feats = self.pin.get_support_features(
            q_pin, v_pin,
            np.array([contacts.left_contact, contacts.right_contact]),
            support_points_world=support_pts if len(support_pts) else None,
        )

        log = StepLog(step=step, t=t)
        log.base_x, log.base_y, log.base_z = (float(x) for x in qpos[:3])
        log.upright_cos = rt.torso_upright_cos()
        log.com_x, log.com_y, log.com_z = (float(x) for x in com)
        log.com_vx, log.com_vy, log.com_vz = (float(x) for x in com_vel)
        log.lin_momentum_norm = float(np.linalg.norm(hg[:3]))
        log.ang_momentum_norm = float(np.linalg.norm(hg[3:]))
        log.momentum_norm = float(np.linalg.norm(hg))
        cp = feats["capture_point"]
        log.cp_x, log.cp_y = float(cp[0]), float(cp[1])
        log.cp_margin = float(feats["cp_margin"])
        log.com_margin = float(feats["com_margin"])
        log.support_area = float(feats["support_area"])
        centre = feats["support_center"]
        if np.all(np.isfinite(centre)):
            self._support_centre = np.asarray(centre, dtype=np.float64).reshape(2)
        log.support_mode = str(feats["support_mode"])
        log.contact_left = int(contacts.left_contact)
        log.contact_right = int(contacts.right_contact)
        log.grf_left = float(contacts.left_force)
        log.grf_right = float(contacts.right_force)
        log.grf_total = float(contacts.total_normal_force)
        log.head_force = rt.geom_group_force(rt.head_geoms)
        log.torso_force = rt.geom_group_force(rt.torso_geoms)
        log.push_force = float(push_force)
        if motion is not None:
            log.motion_id = motion.motion_id
            log.phase, log.phase_label = motion.phase_at(t)
        return log

    def targets(self, log: StepLog, motion: Motion | None, t: float) -> tuple[np.ndarray, Mode]:
        """ModeSwitch decides; returns (PD joint targets, mode)."""
        mode = self.switch.step(_feature_dict(log, log.step))
        if mode is Mode.NOMINAL or not self.protective_response:
            if motion is None:
                return self.rt.default_joint_targets(), mode
            if self._motion_to_act is None:
                self.bind_motion(motion)
            return self._apply_balance(motion.sample(t)[self._motion_to_act], log), mode
        return self.crouch_targets, mode

    def _apply_balance(self, q_cmd: np.ndarray, log: StepLog) -> np.ndarray:
        """Add the CoM-feedback correction to the reference targets."""
        b = self.balance
        if b is None or not b.enabled:
            return q_cmd
        if not (log.contact_left or log.contact_right):
            return q_cmd  # airborne: no ground to push against

        e = np.array([log.com_x, log.com_y]) - self._support_centre
        edot = np.array([log.com_vx, log.com_vy])

        d_pitch = float(np.clip(-(b.k_ankle * e[0] + b.kd_ankle * edot[0]),
                                -b.max_ankle_offset, b.max_ankle_offset))
        d_roll = float(np.clip(b.k_ankle * e[1] + b.kd_ankle * edot[1],
                               -b.max_ankle_offset, b.max_ankle_offset))
        d_hip = float(np.clip(-(b.k_hip * e[1] + b.kd_hip * edot[1]),
                              -b.max_hip_offset, b.max_hip_offset))

        out = q_cmd.copy()
        idx = self.rt.act_index
        for side, contact in (("left", log.contact_left), ("right", log.contact_right)):
            if not contact:
                continue  # only a loaded foot can generate an ankle moment
            out[idx[f"{side}_ankle_pitch_joint"]] += d_pitch
            out[idx[f"{side}_ankle_roll_joint"]] += d_roll
            out[idx[f"{side}_hip_roll_joint"]] += d_hip
        return out

    def reset(self) -> None:
        self.switch.reset()
