"""Reference motion library for the Kalaripayattu controller demo.

Reference source priority (as specified in the brief):

  1. A real retargeted NPZ in data/motions_retargeted/ with the contract
     q_ref[T, nq], dq_ref, contacts, phase, family, fps. `load_retargeted()`
     checks for these first.
  2. Otherwise, the joint-space authored motions defined below.

SCOPE / HONESTY: the three built-in motions are **joint-space authored keyframe
trajectories** -- named joint angles interpolated with a smooth (cosine) profile
and no inverse kinematics solve behind them. They are Kalaripayattu-inspired
postures, not motion-captured or retargeted human data, and they are not learned.
They exist to give the balance experiments a physically plausible reference to
track. Anything derived from them must be labelled accordingly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

RETARGET_DIR = "data/motions_retargeted"

MOTION_IDS = ("horse_stance_hold", "single_leg_front_kick", "trunk_pivot_strike_prep")


@dataclass
class Motion:
    """A reference trajectory in actuator joint space."""

    motion_id: str
    family: str
    fps: float
    joint_names: list[str]
    q_ref: np.ndarray              # [T, n_joints]
    phase: np.ndarray              # [T] in [0, 1]
    phase_label: list[str]         # [T] human-readable phase name
    source: str                    # "authored_keyframe" or "retargeted_npz"
    contacts_ref: np.ndarray | None = None   # [T, 2] expected (left, right)
    notes: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.q_ref.shape[0])

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps

    def dq_ref(self) -> np.ndarray:
        return np.gradient(self.q_ref, 1.0 / self.fps, axis=0)

    def sample(self, t: float) -> np.ndarray:
        """Reference joint targets at time t (seconds), clamped at the ends."""
        idx = int(np.clip(round(t * self.fps), 0, self.n_frames - 1))
        return self.q_ref[idx]

    def phase_at(self, t: float) -> tuple[float, str]:
        idx = int(np.clip(round(t * self.fps), 0, self.n_frames - 1))
        return float(self.phase[idx]), self.phase_label[idx]

    def single_support_window(self) -> tuple[float, float] | None:
        """(t_start, t_end) of the single-support phase, or None if there isn't one."""
        idx = [i for i, lab in enumerate(self.phase_label) if lab == "single_support"]
        if not idx:
            return None
        return idx[0] / self.fps, idx[-1] / self.fps


def _smoothstep(u: np.ndarray) -> np.ndarray:
    """Cosine ease-in/ease-out on u in [0, 1]; zero velocity at both ends."""
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(u, 0.0, 1.0))


def _interp_keyframes(
    keyframes: list[tuple[float, dict[str, float]]],
    joint_names: list[str],
    fps: float,
    base_pose: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothly interpolate a list of (time, {joint: angle}) keyframes."""
    duration = keyframes[-1][0]
    n = int(round(duration * fps)) + 1
    times = np.arange(n) / fps
    idx = {n_: i for i, n_ in enumerate(joint_names)}

    traj = np.zeros((n, len(joint_names)))
    for name, angle in base_pose.items():
        traj[:, idx[name]] = angle

    # Resolve each keyframe into a full joint vector (carrying values forward).
    resolved = []
    current = dict(base_pose)
    for t_key, targets in keyframes:
        current = {**current, **targets}
        vec = np.zeros(len(joint_names))
        for name, angle in current.items():
            vec[idx[name]] = angle
        resolved.append((t_key, vec))

    for k in range(len(resolved) - 1):
        t0, v0 = resolved[k]
        t1, v1 = resolved[k + 1]
        mask = (times >= t0) & (times <= t1)
        if not mask.any():
            continue
        u = (times[mask] - t0) / max(t1 - t0, 1e-9)
        traj[mask] = v0 + (v1 - v0) * _smoothstep(u)[:, None]
    traj[times >= resolved[-1][0]] = resolved[-1][1]
    return traj, times


def _labels(times: np.ndarray, windows: list[tuple[float, float, str]], default: str) -> list[str]:
    out = [default] * len(times)
    for t0, t1, label in windows:
        for i, t in enumerate(times):
            if t0 <= t <= t1:
                out[i] = label
    return out


def build_horse_stance_hold(joint_names: list[str], base_pose: dict[str, float],
                            fps: float = 50.0) -> Motion:
    """Kalari 'vadivu' horse stance: wide, deep, symmetric double support."""
    wide = {
        "left_hip_roll_joint": 0.28, "right_hip_roll_joint": -0.28,
        "left_hip_pitch_joint": -0.52, "right_hip_pitch_joint": -0.52,
        "left_knee_joint": 1.00, "right_knee_joint": 1.00,
        "left_ankle_pitch_joint": -0.50, "right_ankle_pitch_joint": -0.50,
        "left_shoulder_pitch_joint": 0.05, "right_shoulder_pitch_joint": 0.05,
        "left_shoulder_roll_joint": 0.30, "right_shoulder_roll_joint": -0.30,
        "left_elbow_joint": 1.35, "right_elbow_joint": 1.35,
        "waist_pitch_joint": 0.05,
    }
    kf = [(0.0, {}), (1.6, wide), (3.4, wide), (4.6, {k: base_pose.get(k, 0.0) for k in wide})]
    traj, times = _interp_keyframes(kf, joint_names, fps, base_pose)
    labels = _labels(times, [(1.6, 3.4, "hold")], "transition")
    return Motion(
        motion_id="horse_stance_hold", family="stance", fps=fps, joint_names=joint_names,
        q_ref=traj, phase=times / times[-1], phase_label=labels,
        source="authored_keyframe",
        contacts_ref=np.ones((len(times), 2), dtype=bool),
        notes="Symmetric deep double-support stance; descend, hold 1.8 s, return.",
    )


def build_single_leg_front_kick(joint_names: list[str], base_pose: dict[str, float],
                                fps: float = 50.0) -> Motion:
    """Front kick off the right leg: weight shift, single support, kick, recover.

    The single-support window is what the push-recovery sweep perturbs.
    """
    shift = {
        "left_hip_roll_joint": 0.16, "right_hip_roll_joint": 0.10,
        "waist_roll_joint": -0.08,
        "left_hip_pitch_joint": -0.30, "left_knee_joint": 0.55,
        "left_ankle_pitch_joint": -0.28,
    }
    lift = {**shift,
            "right_hip_pitch_joint": -0.70, "right_knee_joint": 1.20,
            "right_ankle_pitch_joint": -0.10,
            "left_shoulder_roll_joint": 0.45, "right_shoulder_roll_joint": -0.45,
            "left_elbow_joint": 1.20, "right_elbow_joint": 1.20}
    extend = {**lift,
              "right_hip_pitch_joint": -1.05, "right_knee_joint": 0.25,
              "right_ankle_pitch_joint": -0.20, "waist_pitch_joint": 0.12}
    retract = {**lift}
    kf = [
        (0.0, {}),
        (1.0, shift),      # shift weight onto the left leg
        (1.8, lift),       # right foot leaves the floor -> single support
        (2.6, extend),     # kick extension
        (3.2, retract),    # retract
        (4.0, shift),      # foot back down
        (5.0, {k: base_pose.get(k, 0.0) for k in extend}),
    ]
    traj, times = _interp_keyframes(kf, joint_names, fps, base_pose)
    labels = _labels(
        times,
        [(0.0, 1.0, "weight_shift"), (1.0, 1.8, "lift_off"),
         (1.8, 3.6, "single_support"), (3.6, 4.2, "touch_down"), (4.2, 5.0, "recover")],
        "transition",
    )
    contacts = np.ones((len(times), 2), dtype=bool)
    contacts[[i for i, l in enumerate(labels) if l == "single_support"], 1] = False
    return Motion(
        motion_id="single_leg_front_kick", family="kick", fps=fps, joint_names=joint_names,
        q_ref=traj, phase=times / times[-1], phase_label=labels,
        source="authored_keyframe", contacts_ref=contacts,
        notes="Right-leg front kick; single support on the left leg from 1.8 s to 3.6 s.",
    )


def build_trunk_pivot_strike_prep(joint_names: list[str], base_pose: dict[str, float],
                                  fps: float = 50.0) -> Motion:
    """Trunk pivot into a strike-ready posture; double support throughout."""
    load = {
        "left_hip_pitch_joint": -0.38, "right_hip_pitch_joint": -0.30,
        "left_knee_joint": 0.72, "right_knee_joint": 0.60,
        "left_ankle_pitch_joint": -0.36, "right_ankle_pitch_joint": -0.32,
        "left_hip_roll_joint": 0.10, "right_hip_roll_joint": -0.10,
    }
    pivot = {**load,
             "waist_yaw_joint": 0.55, "waist_pitch_joint": 0.10, "waist_roll_joint": 0.06,
             "right_shoulder_pitch_joint": -0.45, "right_shoulder_roll_joint": -0.55,
             "right_elbow_joint": 1.55, "left_shoulder_pitch_joint": 0.55,
             "left_shoulder_roll_joint": 0.20, "left_elbow_joint": 1.10}
    strike = {**pivot,
              "waist_yaw_joint": -0.30, "right_shoulder_pitch_joint": 0.30,
              "right_elbow_joint": 0.40, "right_shoulder_roll_joint": -0.20}
    kf = [(0.0, {}), (1.2, load), (2.4, pivot), (3.2, strike), (4.4, load),
          (5.2, {k: base_pose.get(k, 0.0) for k in strike})]
    traj, times = _interp_keyframes(kf, joint_names, fps, base_pose)
    labels = _labels(times, [(1.2, 2.4, "load"), (2.4, 3.2, "pivot"),
                             (3.2, 4.4, "strike_prep")], "transition")
    return Motion(
        motion_id="trunk_pivot_strike_prep", family="strike", fps=fps, joint_names=joint_names,
        q_ref=traj, phase=times / times[-1], phase_label=labels,
        source="authored_keyframe",
        contacts_ref=np.ones((len(times), 2), dtype=bool),
        notes="Waist-yaw wind-up and release into a strike-ready posture, double support.",
    )


BUILDERS = {
    "horse_stance_hold": build_horse_stance_hold,
    "single_leg_front_kick": build_single_leg_front_kick,
    "trunk_pivot_strike_prep": build_trunk_pivot_strike_prep,
}


def load_retargeted(motion_id: str, joint_names: list[str]) -> Motion | None:
    """Load a real retargeted NPZ if one exists, else None (priority 1)."""
    if not os.path.isdir(RETARGET_DIR):
        return None
    for fname in sorted(os.listdir(RETARGET_DIR)):
        if not fname.endswith(".npz") or motion_id not in fname:
            continue
        path = os.path.join(RETARGET_DIR, fname)
        data = np.load(path, allow_pickle=True)
        required = {"q_ref", "dq_ref", "contacts", "phase", "family", "fps"}
        missing = required - set(data.files)
        if missing:
            print(f"[motions] {path} ignored: missing keys {sorted(missing)}")
            continue
        q_ref = np.asarray(data["q_ref"], dtype=np.float64)
        if q_ref.shape[1] == len(joint_names) + 7:
            q_ref = q_ref[:, 7:]  # strip the floating base
        if q_ref.shape[1] != len(joint_names):
            print(f"[motions] {path} ignored: q_ref has {q_ref.shape[1]} columns, "
                  f"expected {len(joint_names)} or {len(joint_names) + 7}")
            continue
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        phase = np.asarray(data["phase"], dtype=np.float64).reshape(-1)
        contacts = np.asarray(data["contacts"]).astype(bool).reshape(len(q_ref), -1)
        print(f"[motions] using REAL retargeted motion {path} ({len(q_ref)} frames @ {fps} fps)")
        return Motion(
            motion_id=motion_id, family=str(data["family"]), fps=fps,
            joint_names=joint_names, q_ref=q_ref, phase=phase,
            phase_label=["retargeted"] * len(q_ref), source="retargeted_npz",
            contacts_ref=contacts, notes=f"Loaded from {path}",
        )
    return None


def get_motion(motion_id: str, joint_names: list[str], base_pose: dict[str, float],
               fps: float = 50.0) -> Motion:
    """Reference-source priority: real retarget first, authored keyframes second."""
    if motion_id not in BUILDERS:
        raise KeyError(f"unknown motion '{motion_id}'; known: {sorted(BUILDERS)}")
    real = load_retargeted(motion_id, joint_names)
    if real is not None:
        return real
    return BUILDERS[motion_id](joint_names, base_pose, fps)
