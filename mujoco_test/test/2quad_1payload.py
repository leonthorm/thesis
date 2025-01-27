import os
import mujoco
import glfw
import numpy as np

dirname = os.path.dirname(__file__)
xml_file_path = dirname + "/../../src/dynamics/bitcraze_crazyflie_2/scene_2quad_payload.xml"

model = mujoco.MjModel.from_xml_path(xml_file_path)
data = mujoco.MjData(model)

# Initialize GLFW
glfw.init()
window = glfw.create_window(800, 600, "MuJoCo Simulation", None, None)
glfw.make_context_current(window)

# Initialize MuJoCo rendering context and scene
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
scene = mujoco.MjvScene(model, maxgeom=1000)

# Camera and options
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
opt = mujoco.MjvOption()  # Visualization options

keyframe_qpos = np.array([0, 0, 0.1, 1, 0, 0, 0])  # Example qpos (position and orientation)
keyframe_ctrl = np.array([0.1, 0.1, 0.1, 0.1,0.1, 0.1, 0.1, 0.1])
# Main rendering loop
while not glfw.window_should_close(window):
    # data.qpos[:] = keyframe_qpos  # Set the position and orientation from keyframe
    print(model.body)
    mujoco.mj_step(model, data)

    # Update the scene
    mujoco.mjv_updateScene(
        model,
        data,
        opt,
        None,  # No perturbation
        cam,
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )

    # Render the scene
    mujoco.mjr_render(mujoco.MjrRect(0, 0, 800, 600), scene, context)

    # Swap buffers and poll for events
    glfw.swap_buffers(window)
    glfw.poll_events()

# Cleanup
glfw.terminate()
