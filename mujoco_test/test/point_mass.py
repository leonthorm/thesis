import os

import mujoco
import numpy as np


if not glfw.init():
    raise Exception("Could not initialize GLFW")

window = glfw.create_window(800, 600, "Point Mass with PID Controller", None, None)
if not window:
    glfw.terminate()
    raise Exception("Could not create GLFW window")

glfw.make_context_current(window)
path = os.getcwd()

xml_file_path = path+"/dynamics/point_mass.xml"
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

Kp = 30.0
Ki = 2.2
Kd = 2.5

target = np.array([1.0, 0.5, 1.0])

integral_error = np.zeros(3)
previous_error = np.zeros(3)
dt = model.opt.timestep

trajectory_data = []
while not glfw.window_should_close(window):
    current_position = data.xpos[model.body("point_mass").id]
    current_velocity = np.array([data.qvel[model.joint("joint_1").id],
                                 data.qvel[model.joint("joint_2").id],
                                 data.qvel[model.joint("joint_3").id]])
    print('##################')
    print(f"Current xpos: {data.qpos}")
    print(f"Current qpos: {data.qpos}")
    #print(f"Current Velocity: {current_velocity}")
    error = target - current_position

    integral_error += error * dt
    derivative_error = (error - previous_error) / dt
    previous_error = error

    #print(error, previous_error, integral_error, derivative_error, dt)

    control_force = Kp * error + Ki * integral_error + Kd * derivative_error
    print(control_force)
    data.ctrl[:] = control_force

    mujoco.mj_step(model, data, nstep=1)

    #state = np.concatenate([current_position, current_velocity])
    state = current_position
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
np.savetxt("mujoco_test/point_mass_states.csv", states, delimiter=",")
np.savetxt("mujoco_test/point_mass_actions.csv", actions, delimiter=",")
