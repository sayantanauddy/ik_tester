import numpy as np
import pytest
import mujoco
from pathlib import Path
from omegaconf import DictConfig
from hydra import initialize, compose
from ik_tester.ik1 import IK1
from ik_tester.utils import get_repo_root
import time
import mediapy as media
import logging

REPO_ROOT = get_repo_root()
MODEL_XML = str(Path(REPO_ROOT / "assets/franka_emika_panda/scene.xml"))
MOCAP_NAME = "target"


def fig8_traj(radius=0.3, center=(0.5, 0.0, 0.5), n_points=100, plane="yz"):
    """
    Generate a figure-8 (Gerono/Lissajous-like) trajectory in the requested plane.
    plane: one of "xy", "xz", "yz".
    radius: amplitude along both axes.
    """

    theta = np.linspace(0, 2 * np.pi, n_points)
    # Gerono-like figure-8: primary = sin(t), secondary = sin(t)*cos(t) (equivalently 0.5*sin(2t))
    secondary = radius * np.sin(theta)
    primary = radius * np.sin(theta) * np.cos(theta)

    traj = np.zeros((n_points, 3))
    if plane == "xy":
        traj[:, 0] = center[0] + primary
        traj[:, 1] = center[1] + secondary
        traj[:, 2] = center[2]
    elif plane == "xz":
        traj[:, 0] = center[0] + primary
        traj[:, 1] = center[1]
        traj[:, 2] = center[2] + secondary
    elif plane == "yz":
        traj[:, 0] = center[0]
        traj[:, 1] = center[1] + primary
        traj[:, 2] = center[2] + secondary
    else:
        raise ValueError("plane must be one of 'xy', 'xz', or 'yz'")

    return traj

def limit_trajectory_velocity(traj: np.ndarray, max_vel: float, dt: float) -> np.ndarray:
    """
    Resample a trajectory to ensure that the distance between consecutive points 
    corresponds to a velocity <= max_vel.
    
    Args:
        traj: (N,3) array of points
        max_vel: maximum allowed velocity [m/s]
        dt: simulation integration timestep [s]

    Returns:
        traj_limited: (M,3) array of points with limited velocity
    """
    traj_limited = [traj[0]]
    
    for i in range(1, len(traj)):
        delta = traj[i] - traj_limited[-1]
        dist = np.linalg.norm(delta)
        max_step = max_vel * dt
        if dist <= max_step:
            traj_limited.append(traj[i])
        else:
            # interpolate intermediate points
            n_steps = int(np.ceil(dist / max_step))
            for j in range(1, n_steps + 1):
                traj_limited.append(traj_limited[-1] + delta / n_steps)
    
    return np.array(traj_limited)


def test_fig8(
        traj_plane="xy", 
        traj_radius=0.3, 
        max_track_vel=0.01, 
        frame_w:int=320, 
        frame_h:int=240,
        ):
    """Track a figure 8 trajectory and return video + metrics."""

    # Load config
    with initialize(config_path="../../config", job_name="test_ik"):
        cfg: DictConfig = compose(config_name="ik1")

    # Load Mujoco model
    model = mujoco.MjModel.from_xml_path(MODEL_XML)
    data = mujoco.MjData(model)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam) 

    # tweak camera to taste
    cam.distance = 2.0                  
    #cam.azimuth = 150
    #cam.elevation = -25

    # Mocap body to track
    mocap_id = model.body(MOCAP_NAME).mocapid[0]

    # IK solver
    ik_solver = IK1(model, data, cfg)

    # Site and joints
    site_id = ik_solver.site_id
    dof_ids = ik_solver.dof_ids
    actuator_ids = ik_solver.actuator_ids
    
    # Reference for joints
    q_ref = model.key(cfg.key_name).qpos

    #Trajectory
    traj = fig8_traj(plane=traj_plane, radius=traj_radius)
    traj = np.concatenate([traj]*3)
    traj = limit_trajectory_velocity(traj, max_vel=max_track_vel, dt=cfg.integration_dt)

    ref_positions, exec_positions = [], []

    frames = []

    #Main simulation loop
    mujoco.mj_resetDataKeyframe(model, data, model.key(cfg.key_name).id)

    # enable joint visualization option:
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

    # For now, we fix the orientation to the initial mocap orientation 
    quat_des = data.mocap_quat[mocap_id]

    with mujoco.Renderer(model, width=frame_w, height=frame_h) as renderer:

        # Wait till EE reaches the start of trajectory
        p_des = traj[0]
        tol = np.array([0.005, 0.005, 0.005])
        max_iters = 1000
        it = 0
        while True:
            it += 1
            q_pos = data.qpos.copy()
            q_next = ik_solver.solve(q_pos, p_des, quat_des, q_ref)
            data.ctrl[actuator_ids] = q_next[dof_ids]
            mujoco.mj_step(model, data)

            renderer.update_scene(data, scene_option=scene_option, camera=cam)
            pixels = renderer.render()
            frames.append(pixels)

            gripper_pos = data.site(site_id).xpos.copy()

            if np.all(np.abs(p_des - gripper_pos) <= tol):
                break
            if it >= max_iters:
                logging.warning(f"Warning: did not reach target within {max_iters} iterations; final abs error = {np.abs(p_des - gripper_pos)}")
                break
            
        # Wait to start tracking
        time.sleep(1)

        # Start tracking
        for p_des in traj:

            # Update the mocap target for visualization
            data.mocap_pos[mocap_id] = p_des

            q_pos = data.qpos.copy()

            #Solve IK
            q_next = ik_solver.solve(q_pos, p_des, quat_des, q_ref)
            data.ctrl[actuator_ids] = q_next[dof_ids]

            #Step simulation
            mujoco.mj_step(model, data)

            #Record positions for metrics
            ref_positions.append(p_des.copy())
            exec_positions.append(data.site(site_id).xpos.copy())

            #Render frame
            renderer.update_scene(data, scene_option=scene_option, camera=cam)
            pixels = renderer.render()
            frames.append(pixels)

    #Compute tracking error
    ref_positions = np.array(ref_positions)
    exec_positions = np.array(exec_positions)

    return ref_positions, exec_positions, frames


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,3)

    ref0, exec0, frames0 = test_fig8(traj_plane="xy")
    ref1, exec1, frames1 = test_fig8(traj_plane="yz")
    ref2, exec2, frames2 = test_fig8(traj_plane="xz")

    ax[0].plot(ref0[:,0],ref0[:,1], color="blue", label="ref")
    ax[0].plot(exec0[:,0],exec0[:,1], color="red", label="exec")

    ax[1].plot(ref1[:,1],ref1[:,2], color="blue", label="ref")
    ax[1].plot(exec1[:,1],exec1[:,2], color="red", label="exec")

    ax[2].plot(ref2[:,0],ref2[:,2], color="blue", label="ref")
    ax[2].plot(exec2[:,0],exec2[:,2], color="red", label="exec")

    plt.show()
