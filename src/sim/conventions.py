"""Explicit MuJoCo <-> Pinocchio state conversions for the Unitree G1.

Three conventions differ between the engines and every one of them is a silent
source of wrong physics if assumed away. This module makes all three explicit:

(a) Free-joint orientation
      MuJoCo  qpos[3:7] = [w, x, y, z]
      Pinocchio FreeFlyer q[3:7] = [x, y, z, w]

(b) Joint ordering
      Never assumed. Both engines are introspected for their joint names and
      per-joint qpos/qvel addresses, and every conversion is routed through a
      name-keyed index map. If the sets of names disagree, construction fails
      loudly rather than producing a silently misaligned state vector.

(c) Free-joint velocity frame
      MuJoCo  qvel[0:3] = base LINEAR velocity in the WORLD frame
              qvel[3:6] = base ANGULAR velocity in the BODY frame
      Pinocchio FreeFlyer v[0:6] = [linear, angular], BOTH in the BODY frame
      So the linear part must be rotated by R^T; the angular part must not.
      Getting this wrong leaves positions correct but corrupts CoM velocity,
      and therefore the capture point and every downstream stability metric.
"""

from __future__ import annotations

import numpy as np

_MJ_FREE_NQ = 7
_MJ_FREE_NV = 6


def quat_wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def quat_xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Rotation matrix R (world <- body) from a [w, x, y, z] quaternion."""
    w, x, y, z = np.asarray(quat, dtype=np.float64).reshape(4)
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class StateConverter:
    """Name-keyed bidirectional state converter between MuJoCo and Pinocchio.

    Built by introspecting both models; nothing about ordering is assumed.
    """

    def __init__(self, mj_model, pin_model):
        import mujoco  # local import so this module is importable without MuJoCo

        self.mj_nq = int(mj_model.nq)
        self.mj_nv = int(mj_model.nv)
        self.pin_nq = int(pin_model.nq)
        self.pin_nv = int(pin_model.nv)

        # --- MuJoCo side: free joint + hinge joints, keyed by name ------------
        free_joints = [
            i for i in range(mj_model.njnt)
            if mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
        ]
        if len(free_joints) != 1:
            raise ValueError(f"expected exactly 1 MuJoCo free joint, found {len(free_joints)}")
        self.mj_free_jnt = free_joints[0]
        if mj_model.jnt_qposadr[self.mj_free_jnt] != 0:
            raise ValueError("MuJoCo free joint is not first in qpos; unsupported layout")

        self.mj_joints: dict[str, tuple[int, int]] = {}  # name -> (qposadr, dofadr)
        for i in range(mj_model.njnt):
            if i == self.mj_free_jnt:
                continue
            if mj_model.jnt_type[i] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(
                    f"joint {mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)} "
                    "is neither free nor hinge; unsupported"
                )
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            self.mj_joints[name] = (int(mj_model.jnt_qposadr[i]), int(mj_model.jnt_dofadr[i]))

        # --- Pinocchio side: root FreeFlyer + 1-dof joints, keyed by name ----
        self.pin_joints: dict[str, tuple[int, int]] = {}
        root_seen = False
        for jid in range(1, pin_model.njoints):
            jmodel = pin_model.joints[jid]
            name = pin_model.names[jid]
            if jmodel.nq == 7 and jmodel.nv == 6:
                if root_seen:
                    raise ValueError("multiple FreeFlyer joints in Pinocchio model")
                root_seen = True
                if jmodel.idx_q != 0:
                    raise ValueError("Pinocchio FreeFlyer is not first in q; unsupported layout")
                self.pin_root_name = name
                continue
            if jmodel.nq != 1 or jmodel.nv != 1:
                raise ValueError(f"Pinocchio joint {name} has nq={jmodel.nq}, nv={jmodel.nv}")
            self.pin_joints[name] = (int(jmodel.idx_q), int(jmodel.idx_v))
        if not root_seen:
            raise ValueError("no FreeFlyer joint found in Pinocchio model")

        # --- Name reconciliation: fail loudly, never silently truncate -------
        mj_names = set(self.mj_joints)
        pin_names = set(self.pin_joints)
        if mj_names != pin_names:
            only_mj = sorted(mj_names - pin_names)
            only_pin = sorted(pin_names - mj_names)
            raise ValueError(
                "MuJoCo and Pinocchio joint name sets differ.\n"
                f"  only in MuJoCo   ({len(only_mj)}): {only_mj}\n"
                f"  only in Pinocchio({len(only_pin)}): {only_pin}"
            )
        self.joint_names = sorted(mj_names)
        self.n_joints = len(self.joint_names)

        # Vectorised index arrays, aligned on self.joint_names.
        self.mj_qadr = np.array([self.mj_joints[n][0] for n in self.joint_names], dtype=int)
        self.mj_vadr = np.array([self.mj_joints[n][1] for n in self.joint_names], dtype=int)
        self.pin_qadr = np.array([self.pin_joints[n][0] for n in self.joint_names], dtype=int)
        self.pin_vadr = np.array([self.pin_joints[n][1] for n in self.joint_names], dtype=int)

        # True iff both engines happen to already agree on ordering. Reported by
        # the Stage-1 gate for transparency; correctness never depends on it.
        self.orders_match = bool(
            np.array_equal(np.argsort(self.mj_qadr), np.argsort(self.pin_qadr))
        )

    # ------------------------------------------------------------------ q ---
    def mj_to_pin_q(self, q_mj: np.ndarray) -> np.ndarray:
        q_mj = np.asarray(q_mj, dtype=np.float64).reshape(-1)
        if q_mj.shape[0] != self.mj_nq:
            raise ValueError(f"q_mj has {q_mj.shape[0]} entries, expected {self.mj_nq}")
        q_pin = np.zeros(self.pin_nq, dtype=np.float64)
        q_pin[0:3] = q_mj[0:3]
        q_pin[3:7] = quat_wxyz_to_xyzw(q_mj[3:7])
        q_pin[self.pin_qadr] = q_mj[self.mj_qadr]
        return q_pin

    def pin_to_mj_q(self, q_pin: np.ndarray) -> np.ndarray:
        q_pin = np.asarray(q_pin, dtype=np.float64).reshape(-1)
        if q_pin.shape[0] != self.pin_nq:
            raise ValueError(f"q_pin has {q_pin.shape[0]} entries, expected {self.pin_nq}")
        q_mj = np.zeros(self.mj_nq, dtype=np.float64)
        q_mj[0:3] = q_pin[0:3]
        q_mj[3:7] = quat_xyzw_to_wxyz(q_pin[3:7])
        q_mj[self.mj_qadr] = q_pin[self.pin_qadr]
        return q_mj

    # ------------------------------------------------------------------ v ---
    def mj_to_pin_v(self, q_mj: np.ndarray, v_mj: np.ndarray) -> np.ndarray:
        """Convert MuJoCo qvel to Pinocchio v (needs q_mj for the base rotation)."""
        q_mj = np.asarray(q_mj, dtype=np.float64).reshape(-1)
        v_mj = np.asarray(v_mj, dtype=np.float64).reshape(-1)
        if v_mj.shape[0] != self.mj_nv:
            raise ValueError(f"v_mj has {v_mj.shape[0]} entries, expected {self.mj_nv}")
        rot = quat_wxyz_to_matrix(q_mj[3:7])
        v_pin = np.zeros(self.pin_nv, dtype=np.float64)
        v_pin[0:3] = rot.T @ v_mj[0:3]   # world -> body: convention (c)
        v_pin[3:6] = v_mj[3:6]           # already body-frame in both engines
        v_pin[self.pin_vadr] = v_mj[self.mj_vadr]
        return v_pin

    def pin_to_mj_v(self, q_mj: np.ndarray, v_pin: np.ndarray) -> np.ndarray:
        q_mj = np.asarray(q_mj, dtype=np.float64).reshape(-1)
        v_pin = np.asarray(v_pin, dtype=np.float64).reshape(-1)
        if v_pin.shape[0] != self.pin_nv:
            raise ValueError(f"v_pin has {v_pin.shape[0]} entries, expected {self.pin_nv}")
        rot = quat_wxyz_to_matrix(q_mj[3:7])
        v_mj = np.zeros(self.mj_nv, dtype=np.float64)
        v_mj[0:3] = rot @ v_pin[0:3]
        v_mj[3:6] = v_pin[3:6]
        v_mj[self.mj_vadr] = v_pin[self.pin_vadr]
        return v_mj

    # -------------------------------------------------------------- joints ---
    def joint_vector_from_mj(self, q_mj: np.ndarray) -> np.ndarray:
        """Actuated joint angles, ordered by self.joint_names."""
        return np.asarray(q_mj, dtype=np.float64).reshape(-1)[self.mj_qadr]

    def describe(self) -> str:
        return (
            f"StateConverter: {self.n_joints} actuated joints matched by name; "
            f"mj nq/nv={self.mj_nq}/{self.mj_nv}, pin nq/nv={self.pin_nq}/{self.pin_nv}; "
            f"native joint orders {'match' if self.orders_match else 'DIFFER (remapped)'}"
        )
