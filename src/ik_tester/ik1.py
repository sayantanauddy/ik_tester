import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
from omegaconf import DictConfig
import hydra
from abc import ABC, abstractmethod

from ik_tester.utils import get_repo_root
from ik_tester.ik import IKSolver

class IK1(IKSolver):
    """
    Numerical IK solver for the Panda robot using damped least squares.
    Adapted from https://github.com/kevinzakka/mjctrl/blob/main/diffik_nullspace.py
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, cfg: DictConfig):
        self.model = model
        self.data = data
        self.cfg = cfg

        self.joint_names = cfg.joint_names
        self.actuator_names = cfg.actuator_names
        self.dof_ids = np.array([model.joint(name).id for name in self.joint_names])
        self.actuator_ids = np.array([model.actuator(name).id for name in self.actuator_names])
        self.site_id = model.site(cfg.site_name).id
        self.Kn = np.asarray(cfg.Kn)
        self.diag = cfg.damping * np.eye(6)
        self.eye = np.eye(model.nv)

        # Workspace arrays
        self.twist = np.zeros(6)
        self.site_quat = np.zeros(4)
        self.site_quat_conj = np.zeros(4)
        self.error_quat = np.zeros(4)
        self.jac = np.zeros((6, model.nv))

    def solve(
        self,
        q_pos: np.ndarray,
        p_des: np.ndarray,
        quat_des: np.ndarray,
        q_ref: np.ndarray,
    ) -> np.ndarray:

        # Compute twist
        dx = p_des - self.data.site(self.site_id).xpos
        self.twist[:3] = self.cfg.Kpos * dx / self.cfg.integration_dt

        mujoco.mju_mat2Quat(self.site_quat, self.data.site(self.site_id).xmat)
        mujoco.mju_negQuat(self.site_quat_conj, self.site_quat)
        mujoco.mju_mulQuat(self.error_quat, quat_des, self.site_quat_conj)
        mujoco.mju_quat2Vel(self.twist[3:], self.error_quat, 1.0)
        self.twist[3:] *= self.cfg.Kori / self.cfg.integration_dt

        # Jacobian
        mujoco.mj_jacSite(self.model, self.data, self.jac[:3], self.jac[3:], self.site_id)

        # Damped least squares
        dq = self.jac.T @ np.linalg.solve(self.jac @ self.jac.T + self.diag, self.twist)

        # Nullspace control towards reference
        dq += (self.eye - np.linalg.pinv(self.jac) @ self.jac) @ (self.Kn * (q_ref - q_pos[self.dof_ids]))

        # Clamp max velocity
        dq_abs_max = np.abs(dq).max()
        if dq_abs_max > self.cfg.max_angvel:
            dq *= self.cfg.max_angvel / dq_abs_max

        # Integrate to get next joint positions
        q_next = q_pos.copy()
        mujoco.mj_integratePos(self.model, q_next, dq, self.cfg.integration_dt)
        np.clip(q_next, *self.model.jnt_range.T, out=q_next)

        return q_next


# Get repo root dynamically
REPO_ROOT = get_repo_root()


@hydra.main(version_base=None, config_path=f"{REPO_ROOT}/config", config_name="ik1")
def main(cfg: DictConfig):
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load model
    model = mujoco.MjModel.from_xml_path(str(Path(REPO_ROOT / "assets") / cfg.xml_path))
    data = mujoco.MjData(model)

    # Gravity compensation
    model.body_gravcomp[:] = float(cfg.gravity_compensation)
    model.opt.timestep = cfg.dt

    # IDs
    mocap_id = model.body(cfg.mocap_name).mocapid[0]
    q_ref = model.key(cfg.key_name).qpos

    # Initialize IK solver
    ik_solver = IK1(model, data, cfg)

    try:

        with mujoco.viewer.launch_passive(
            model=model,
            data=data,
            show_left_ui=False,
            show_right_ui=False
        ) as viewer:

            mujoco.mj_resetDataKeyframe(model, data, model.key(cfg.key_name).id)
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE


            while viewer.is_running():
                step_start = time.time()

                q_pos = data.qpos.copy()
                p_des = data.mocap_pos[mocap_id]
                quat_des = data.mocap_quat[mocap_id]

                # Compute IK
                q_next = ik_solver.solve(q_pos, p_des, quat_des, q_ref)

                # Apply to actuators
                data.ctrl[ik_solver.actuator_ids] = q_next[ik_solver.dof_ids]

                # Step simulation
                mujoco.mj_step(model, data)
                viewer.sync()

                # Maintain fixed simulation timestep
                sleep_time = cfg.dt - (time.time() - step_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("Ctrl-C pressed. Exiting cleanly...")


if __name__ == "__main__":
    main()
