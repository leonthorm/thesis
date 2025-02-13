import os
import time

import mujoco
import mujoco.viewer
import numpy as np

dirname = os.path.dirname(__file__)
model_path = dirname + "/dynamics/cf2/scene_cf2_cable.xml"
model_path = dirname + "/mujoco/scene_payload.xml"
m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)


keyframe_ctrl = np.array([0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08])
d.ctrl[:] = keyframe_ctrl
with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  viewer.cam.distance = 3.0
  start = time.time()
  while viewer.is_running():
    step_start = time.time()

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    # print(d.ctrl)
    mujoco.mj_step(m, d)

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
