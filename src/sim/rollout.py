"""Shared rollout driver + CSV/video writers used by all Stage 2/3 experiments."""

from __future__ import annotations

import csv
import dataclasses
import os
from dataclasses import dataclass

import numpy as np

from src.sim.controller import KalariController, StepLog
from src.sim.motions import Motion
from src.sim.mujoco_runtime import G1MujocoRuntime


@dataclass
class Push:
    """A lateral pelvis push applied through data.xfrc_applied."""

    t_start: float
    duration: float
    force_y: float
    force_x: float = 0.0

    def active(self, t: float) -> bool:
        return self.t_start <= t < self.t_start + self.duration


@dataclass
class RolloutResult:
    logs: list[StepLog]
    frames: list[np.ndarray]
    transitions: list[dict]
    fell: bool
    final_base_z: float
    min_base_z: float
    peak_head_force: float
    peak_torso_force: float
    head_impulse: float
    torso_impulse: float
    failure_frame: int | None
    failure_time: float | None


def run_rollout(
    rt: G1MujocoRuntime,
    ctrl: KalariController,
    motion: Motion | None,
    duration: float,
    push: Push | None = None,
    record_video: bool = False,
    camera: str = "track",
    video_fps: int = 30,
    width: int = 640,
    height: int = 480,
    fall_height: float = 0.35,
    initial_settle: float = 0.0,
) -> RolloutResult:
    """Run the closed loop for `duration` seconds and collect everything."""
    if motion is not None:
        ctrl.bind_motion(motion)
    ctrl.reset()

    if initial_settle > 0:
        rt.settle(initial_settle)

    n_ticks = int(round(duration * rt.CONTROL_HZ))
    frame_every = max(1, int(round(rt.CONTROL_HZ / video_fps)))
    dt_tick = 1.0 / rt.CONTROL_HZ

    logs: list[StepLog] = []
    frames: list[np.ndarray] = []
    head_impulse = torso_impulse = 0.0
    failure_frame = None
    failure_time = None

    xfrc = np.zeros_like(rt.data.xfrc_applied)

    for k in range(n_ticks):
        t = k * dt_tick

        push_f = 0.0
        xfrc[:] = 0.0
        if push is not None and push.active(t):
            push_f = push.force_y
            xfrc[rt.pelvis_body, 0] = push.force_x
            xfrc[rt.pelvis_body, 1] = push.force_y

        log = ctrl.measure(k, t, motion, push_force=push_f)
        q_cmd, mode = ctrl.targets(log, motion, t)
        log.mode = mode.value

        if motion is not None:
            q_now = rt.data.qpos[rt.act_qadr]
            log.tracking_err_rms = float(np.sqrt(np.mean((q_cmd - q_now) ** 2)))

        tau = rt.control_step(q_cmd, xfrc=xfrc)
        log.torque_norm = float(np.linalg.norm(tau))
        log.torque_max = float(np.max(np.abs(tau)))
        logs.append(log)

        head_impulse += log.head_force * dt_tick
        torso_impulse += log.torso_force * dt_tick

        if failure_frame is None and rt.base_height < fall_height:
            failure_frame = k
            failure_time = t

        if record_video and k % frame_every == 0:
            frames.append(rt.render_frame(camera=camera, width=width, height=height))

    heights = [l.base_z for l in logs]
    return RolloutResult(
        logs=logs,
        frames=frames,
        transitions=ctrl.switch.transition_log,
        fell=bool(logs[-1].base_z < fall_height or min(heights) < fall_height),
        final_base_z=float(logs[-1].base_z),
        min_base_z=float(min(heights)),
        peak_head_force=float(max(l.head_force for l in logs)),
        peak_torso_force=float(max(l.torso_force for l in logs)),
        head_impulse=float(head_impulse),
        torso_impulse=float(torso_impulse),
        failure_frame=failure_frame,
        failure_time=failure_time,
    )


CSV_FIELDS = [f.name for f in dataclasses.fields(StepLog)]


def write_step_csv(path: str, logs: list[StepLog], extra: dict | None = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    extra = extra or {}
    fields = CSV_FIELDS + sorted(extra)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for log in logs:
            row = dataclasses.asdict(log)
            row.update(extra)
            writer.writerow(row)


def write_video(path: str, frames: list[np.ndarray], fps: int = 30) -> bool:
    if not frames:
        print(f"[video] no frames captured, skipping {path}")
        return False
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        import imageio.v2 as imageio

        imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
        print(f"[video] wrote {path} ({len(frames)} frames @ {fps} fps)")
        return True
    except Exception as exc:  # pragma: no cover
        print(f"[video] FAILED to write {path}: {exc}")
        return False
