import numpy as np
import yaml


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


def load_coltans_traj(traj_file, n_robots, dt):
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


def get_coltrans_state_components(traj_file, n_robots, dt, cable_lengths):
    with open(traj_file, 'r') as file:
        trajectory = yaml.safe_load(file)['result']

    ts = np.arange(0, len(trajectory['states']) * dt, dt)
    actions = np.array(trajectory['actions'])
    states = np.array(trajectory['states'])

    payload_pos = states[:, 0:3]
    payload_vel = states[:, 3:6]
    cable_direction = []
    cable_ang_vel = []
    robot_pos = []
    robot_vel = []
    robot_rot = []
    robot_body_ang_vel = []

    for robot in range(n_robots):
        directions = states[:, 6 + 13 * robot:9 + 13 * robot]
        cable_direction.append(directions)
        angular_vel = states[:, 9 + 13 * robot:12 + 13 * robot]
        cable_ang_vel.append(angular_vel)
        # todo: correct?
        positon = payload_pos - cable_lengths[robot] * directions
        robot_pos.append(positon)
        velocity = payload_vel - cable_lengths[robot] * np.cross(angular_vel, directions)
        robot_vel.append(velocity)

        robot_rot.append(states[:, 12 + 13 * robot:16 + 13 * robot])
        robot_body_ang_vel.append(states[:, 16 + 13 * robot:19 + 13 * robot])
    cable_direction = np.array(cable_direction)
    cable_ang_vel = np.array(cable_ang_vel)
    robot_pos = np.array(robot_pos)
    robot_vel = np.array(robot_vel)
    robot_rot = np.array(robot_rot)
    robot_body_ang_vel = np.array(robot_body_ang_vel)

    return ts, payload_pos, payload_vel, cable_direction, cable_ang_vel, robot_rot, robot_pos, robot_body_ang_vel, robot_vel, actions
