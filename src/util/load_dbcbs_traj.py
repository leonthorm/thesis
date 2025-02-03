import numpy as np
import yaml


def load_dbcbs_traj(traj_file, n_robots):
    with open(traj_file, 'r') as file:
        trajectories = yaml.safe_load(file)['result']

    positions, velocities,  actions_array, num_states = [], [], [], []
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
        pos = states[:,0:3]
        vel = states[:,3:6]
        positions.append(pos)
        velocities.append(vel)

    return np.array(positions), np.array(velocities), np.array(actions_array)

def get_dbcbs_trajectory(traj_file, n_robots, dt):

    pos_d, vel_d, acc_d = load_dbcbs_traj(traj_file, n_robots)

    ts = []
    # add 0 action for step 0
    acc_d = np.insert(acc_d, 0, np.array([0.0,0.0,0.0]), axis=1)
    for robot in range(n_robots):

        ti = np.arange(0,len(pos_d[robot]) * dt ,dt)
        assert len(ti) == len(pos_d[robot])
        ts.append(ti)

        # calculate desired velocity
        # for i in range(1, len(acc_d[robot])):
        #
        #     vel_d_i = vel_d[robot][i - 1] + acc_d[robot][i - 1] * dt
        #     vel_d[robot].append(vel_d_i)


    return np.array(ts), pos_d,  vel_d, acc_d
