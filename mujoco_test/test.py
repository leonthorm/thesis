import os
import time

import mujoco
import mujoco.viewer
import numpy as np

from src.util.load_traj import get_coltrans_state_components

dirname = os.path.dirname(__file__)
# model_path = dirname + "/dynamics/cf2/scene_cf2_cable.xml"


expert_traj_dir = dirname + "/../trajectories/expert_trajectories/coltrans_planning"
dynamics_dir = dirname + "/../src/dynamics/"
empty_1robots = expert_traj_dir + "/empty_1robots.yaml"

_, _, _, _, _, _, _, _, _, actions = get_coltrans_state_components(empty_1robots, 1, 0.01, [0.5])

model_path = dirname + "/catenary.xml"
model_path = dirname + "/single_quad_payload.xml"
m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)
keyframe_ctrl = np.array([0.0845] * 8)
# d.ctrl[:] = keyframe_ctrl
t = 0

warmstart = True
with mujoco.viewer.launch_passive(m, d) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    viewer.cam.distance = 5.0
    start = time.time()

    while viewer.is_running():
        if warmstart:
            for _ in range(300):
                step_start = time.time()
                mujoco.mj_step(m, d)
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
            d.eq_active[:] = 0
            warmstart = False
            print("deactivated")

        step_start = time.time()
        d.ctrl[:] = actions[t]
        d.ctrl[:] = np.array([1.5, 0, 1.5, 0]) * (0.0356 * 9.81 / 4.)
        t += 1
        print(d.ctrl)
        step_start = time.time()
        if t > 400:
            t = 0

        mujoco.mj_step(m, d)


        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
