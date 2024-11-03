import mujoco
import glfw
import numpy as np
from matplotlib.pyplot import axline

if not glfw.init():
    raise Exception("Could not initialize GLFW")

window = glfw.create_window(800, 600, "MuJoCo Point Mass with PID Controller", None, None)
if not window:
    glfw.terminate()
    raise Exception("Could not create GLFW window")

glfw.make_context_current(window)

xml_file_path = "/home/simba/projects/thesis/mujoco_test/point_mass.xml"  # Update with your actual file path
model = mujoco.MjModel.from_xml_path(xml_file_path)
data = mujoco.MjData(model)

context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
scene = mujoco.MjvScene(model, maxgeom=1000)
camera = mujoco.MjvCamera()
option = mujoco.MjvOption()

camera.azimuth = 90
camera.elevation = -20
camera.distance = 3
camera.lookat[:] = [0, 0, 0]

Kp = 10.0
Ki = 0.5
Kd = 1.0

setpoint = np.array([1.0, 0.0, 1.0])  # Example target position

integral_error = np.zeros(3)
previous_error = np.zeros(3)
dt = model.opt.timestep  # Time step

trajectory_data = []
while not glfw.window_should_close(window):
    current_position = data.xpos[model.body("point_mass").id]
    current_velocity = np.array([data.qvel[model.joint("joint_1").id],  # Velocity for x-axis
                                 data.qvel[model.joint("joint_2").id],  # Velocity for y-axis
                                 data.qvel[model.joint("joint_3").id]])  # Velocity for z-axis

    print(f"Current Position: {current_position}, Shape: {current_position.shape}")
    print(f"Current Velocity: {current_velocity}, Shape: {current_velocity.shape}")
    error = setpoint - current_position

    integral_error += error * dt
    derivative_error = (error - previous_error) / dt
    previous_error = error

    control_force = Kp * error + Ki * integral_error + Kd * derivative_error

    data.xfrc_applied[model.body("point_mass").id][:3] = control_force

    state = np.concatenate([current_position, current_velocity])
    action = control_force
    trajectory_data.append((state, action))

    mujoco.mj_step(model, data)

    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
    mujoco.mjr_render(viewport, scene, context)

    mujoco.mjv_updateScene(model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()

states, actions = zip(*trajectory_data)
states = np.array(states)
actions = np.array(actions)

print(states, actions)