import yaml
import numpy as np


def saveyaml(file_out, data):
    with open(file_out, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=None)


def loadyaml(file_in):
    with open(file_in, "r") as f:
        file_out = yaml.safe_load(f)
    return file_out


def loadcsv(filename):
    return np.loadtxt(filename, delimiter=",", skiprows=1, ndmin=2)


def deconstruct_obs(obs, num_robots):
    i = obs[-1]

    # if self.payloadType == "point":
    cable_start_idx = 9
    robot_start_idx = cable_start_idx + 6
    other_robot_pos_start_idx = robot_start_idx + 13

    payload_state_d_start_idx = other_robot_pos_start_idx + (num_robots - 1) * 3
    cable_state_d_start_idx = payload_state_d_start_idx + 6
    robot_state_d_start_idx = cable_state_d_start_idx + 6
    actions_d_start_idx = robot_state_d_start_idx + 13

    payload_pos = obs[0: 3]
    payload_vel = obs[3: 6]
    payload_acc = obs[6: 9]

    qc = obs[cable_start_idx: cable_start_idx + 3]
    wc = obs[cable_start_idx + 3: cable_start_idx + 6]
    quat = obs[robot_start_idx: robot_start_idx + 4]
    pos = obs[robot_start_idx + 4: robot_start_idx + 7]
    w = obs[robot_start_idx + 7: robot_start_idx + 10]
    vel = obs[robot_start_idx + 10: robot_start_idx + 13]

    other_robot_pos = obs[other_robot_pos_start_idx: other_robot_pos_start_idx + 3]

    payload_pos_d = obs[payload_state_d_start_idx: payload_state_d_start_idx + 3]
    payload_vel_d = obs[payload_state_d_start_idx + 3: payload_state_d_start_idx + 6]

    qc_d = obs[cable_state_d_start_idx: cable_state_d_start_idx + 3]
    wc_d = obs[cable_state_d_start_idx + 3: cable_state_d_start_idx + 6]
    quat_d = obs[robot_state_d_start_idx: robot_state_d_start_idx + 4]
    pos_d = obs[robot_state_d_start_idx + 4: robot_state_d_start_idx + 7]
    w_d = obs[robot_state_d_start_idx + 7: robot_state_d_start_idx + 10]
    vel_d = obs[robot_state_d_start_idx + 10: robot_state_d_start_idx + 13]

    actions_d = obs[actions_d_start_idx: actions_d_start_idx + 4]

    return (payload_pos, payload_vel, payload_acc,
            qc, wc,
            quat, pos, w, vel,
            other_robot_pos,
            payload_pos_d, payload_vel_d,
            qc_d, wc_d,
            quat_d, pos_d, w_d, vel_d,
            actions_d,
            i)


def derivative(vec, dt):
    dvec = []
    # dvec  =[[0,0,0]]
    for i in range(len(vec) - 1):
        dvectmp = (vec[i + 1] - vec[i]) / dt
        dvec.append(dvectmp.tolist())
    dvec.append([0, 0, 0])
    return np.asarray(dvec)


def calculate_observation_space_size_old(num_robots):
    payload_pos, payload_vel, payload_acc = 3, 3, 3
    cable_direction, cable_force = 3, 3
    robot_pos, robot_vel, robot_rot, robot_body_ang_vel = 3, 3, 4, 3
    other_robot_pos = 3
    action_d = 4
    robot_id = 1

    size = (payload_acc + (payload_pos + payload_vel
                           + cable_direction + cable_force
                           + robot_pos + robot_vel + robot_rot + robot_body_ang_vel) * 2
            + (num_robots - 1) * other_robot_pos +
            action_d
            + robot_id)

    return size


def calculate_observation_space_size(num_robots):
    state_size = (6 + 6 * num_robots + 7 * num_robots)
    state_d_size = state_size + 3
    acc_d_size = 3
    action_size = 4 * num_robots
    observation_space_size = state_size + state_d_size + acc_d_size + action_size

    return observation_space_size


def split_observation(obs, num_robots):
    state_size = (6 + 6 * num_robots + 7 * num_robots)
    state_d_start_idx = state_size
    acc_d_start_idx = state_d_start_idx + state_size + 3
    action_d_start_idx = acc_d_start_idx + 3

    state = obs[:state_d_start_idx]
    state_d = obs[state_d_start_idx:acc_d_start_idx]
    acc_d = obs[acc_d_start_idx:action_d_start_idx]
    actions_d = obs[action_d_start_idx:]

    return np.array(state), np.array(state_d), np.array(acc_d), np.array(actions_d)


def calculate_state_size(num_robots):
    payload_pos, payload_vel = 3, 3
    cable_direction, cable_force = 3, 3
    robot_pos, robot_vel, robot_rot, robot_body_ang_vel = 3, 3, 4, 3
    other_robot_pos = 3
    action_d = 4
    robot_id = 1

    size = (payload_pos + payload_vel
            + (cable_direction + cable_force) * num_robots
            + (robot_rot + robot_body_ang_vel) * num_robots
            )

    return size


def reconstruct_coltrans_state(full_state_obs):
    full_state_obs = full_state_obs.numpy()
    num_robots = full_state_obs.shape[0]
    state_length = calculate_state_size(num_robots)
    state = np.zeros(state_length)
    state_d = np.zeros(state_length)
    actions_d_all = np.zeros(4 * num_robots)
    acc = np.zeros(3)
    cable_start_idx = 6
    robot_start_idx = cable_start_idx + num_robots * 6

    for robot in range(num_robots):
        (payload_pos, payload_vel, payload_acc,
         qc, wc,
         quat, pos, w, vel,
         other_robot_pos,
         payload_pos_d, payload_vel_d,
         qc_d, wc_d,
         quat_d, pos_d, w_d, vel_d,
         actions_d,
         i) = deconstruct_obs(full_state_obs[robot], num_robots)

        if robot == 0:
            state[0:3] = payload_pos
            state_d[0:3] = payload_pos_d

            state[3:6] = payload_vel
            state_d[3:6] = payload_vel_d

            acc = payload_acc

        state[cable_start_idx + 6 * robot: cable_start_idx + 6 * robot + 3] = qc
        state_d[cable_start_idx + 6 * robot: cable_start_idx + 6 * robot + 3] = qc_d

        state[cable_start_idx + 6 * robot + 3: cable_start_idx + 6 * robot + 6] = wc
        state_d[cable_start_idx + 6 * robot + 3: cable_start_idx + 6 * robot + 6] = wc_d

        state[robot_start_idx + 7 * robot: robot_start_idx + 7 * robot + 4] = quat
        state_d[robot_start_idx + 7 * robot: robot_start_idx + 7 * robot + 4] = quat_d

        state[robot_start_idx + 7 * robot + 4: robot_start_idx + 7 * robot + 7] = w
        state_d[robot_start_idx + 7 * robot + 4: robot_start_idx + 7 * robot + 7] = w_d

        actions_d_all[4 * robot: 4 * robot + 4] = actions_d / (0.0356 * 9.81 / 4.)

    return state, state_d, actions_d_all, acc
