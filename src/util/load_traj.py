import numpy as np
import yaml

from src.util.helper import calculate_state_size


def load_dbcbs_traj(traj_file, n_robots):
    with open(traj_file, 'r') as file:
        trajectories = yaml.safe_load(file)['result']

    positions, velocities, actions_array, num_states = [], [], [], []
    assert len(trajectories) == n_robots
    max_traj_length = -1
    for trajectory in trajectories:
        if max_traj_length < trajectory['num_states']:
            max_traj_length = trajectory['num_states']
    for trajectory in trajectories:

        actions = np.array(trajectory['actions'])
        states = np.array(trajectory['states'])

        original_length = len(states)
        if original_length < max_traj_length:
            last_state = states[-1]
            stay_in_place = [0.0, 0.0, 0.0]
            last_state[3:6] = stay_in_place
            to_append = [last_state] * (max_traj_length - original_length)
            states = np.concatenate((states, to_append), axis=0)

            to_append = [stay_in_place] * (max_traj_length - original_length)
            actions = np.concatenate((actions, to_append), axis=0)
        actions_array.append(actions)
        pos = states[:, 0:3]
        vel = states[:, 3:6]
        positions.append(pos)
        velocities.append(vel)

    return np.array(positions), np.array(velocities), np.array(actions_array)


def get_dbcbs_trajectory(traj_file, n_robots, dt):
    pos_d, vel_d, acc_d = load_dbcbs_traj(traj_file, n_robots)

    ts = []
    # add 0 action for step 0
    acc_d = np.insert(acc_d, 0, np.array([0.0, 0.0, 0.0]), axis=1)
    for robot in range(n_robots):
        ti = np.arange(0, len(pos_d[robot]) * dt, dt)
        assert len(ti) == len(pos_d[robot])
        ts.append(ti)

        # calculate desired velocity
        # for i in range(1, len(acc_d[robot])):
        #
        #     vel_d_i = vel_d[robot][i - 1] + acc_d[robot][i - 1] * dt
        #     vel_d[robot].append(vel_d_i)

    return np.array(ts), pos_d, vel_d, acc_d


def load_coltans_traj_and_split(traj_file, n_robots, dt):
    with open(traj_file, 'r') as file:
        trajectory = yaml.safe_load(file)['result']

    ts = np.arange(0, len(trajectory['states']) * dt, dt)
    actions = np.array(trajectory['actions'])
    states = np.array(trajectory['states'])

    payload_states = states[:, 0:6]
    cable_states = states[:, 6:12]
    robot_states = states[:, 12:]
    assert robot_states.shape == (len(states), n_robots * 13)
    return ts, payload_states, cable_states, robot_states, actions

def load_coltans_traj(traj_file):
    with open(traj_file, 'r') as file:
        refresult = yaml.safe_load(file)['result']
    if "states" in refresult:
        refstate = refresult["states"]

    elif "result" in refresult:
        refstate = refresult["result"]["states"]
        payloadType = refresult["result"]["payload"]
    else:
        raise NotImplementedError("unknown result format")
    if "actions" in refresult:
        refactions = refresult["actions"]
    elif "result" in refresult:
        refactions = refresult["result"]["actions"]
    else:
        raise NotImplementedError("unknown result format")


    return refresult


def get_coltrans_state_components(traj_file, n_robots, dt, cable_lengths):
    with open(traj_file, 'r') as file:
        trajectory = yaml.safe_load(file)['result']

    ts = np.arange(0, len(trajectory['states']) * dt, dt)
    actions = np.array(trajectory['actions_d'])
    # unnormalize
    actions *= (0.0356 * 9.81 / 4.)
    states = np.array(trajectory['states'])

    payload_pos = states[:, 0:3]
    payload_vel = states[:, 3:6]
    cable_direction = []
    cable_ang_vel = []
    robot_pos = []
    robot_vel = []
    robot_rot = []
    robot_body_ang_vel = []

    robot_st_start = 6 + 6 * (n_robots-1) + 6
    for robot in range(n_robots):
        cable_st = states[:, 6 + 6 * robot: 6 + 6 * robot + 6]
        q_cables = cable_st[:, 0:3]
        cable_direction.append(q_cables)
        w_cables = cable_st[:, 3:6]
        cable_ang_vel.append(w_cables)
        # todo: correct?
        positon = payload_pos - cable_lengths[robot] * q_cables
        robot_pos.append(positon)
        velocity = payload_vel - cable_lengths[robot] * np.cross(w_cables, q_cables)
        robot_vel.append(velocity)

        robot_st = states[:, robot_st_start + 7 * robot: robot_st_start + 7 * robot + 7]
        robot_rot.append(robot_st[:, 0:4])
        robot_body_ang_vel.append(robot_st[:, 4:7])
    cable_direction = np.array(cable_direction)
    cable_ang_vel = np.array(cable_ang_vel)
    robot_pos = np.array(robot_pos)
    robot_vel = np.array(robot_vel)
    robot_rot = np.array(robot_rot)
    robot_body_ang_vel = np.array(robot_body_ang_vel)

    return ts, payload_pos, payload_vel, cable_direction, cable_ang_vel, robot_rot, robot_pos, robot_body_ang_vel, robot_vel, actions


def load_model(model_path):
    with open(model_path, "r") as f:
        model = yaml.safe_load(f)
    num_robots = model["num_robots"]

    return model, num_robots


