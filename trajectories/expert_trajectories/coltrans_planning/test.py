import os

import numpy as np
import yaml

dirname = os.path.dirname(__file__)
traj_dir = dirname + "/../trajectories/expert_trajectories/coltrans_planning/"

dynamics = dirname + "/../src/dynamics/"

trajectory_opt = dirname + "/forest_4robots.yaml"

with open(trajectory_opt, 'r') as file:
    cable_lengths = [0.5,0.5,0.5,0.5]
    trajectory = yaml.safe_load(file)['result']
    actions = np.array(trajectory['actions'])
    states = np.array(trajectory['states'])
    n_robots = int(len(actions[0])/4)

    payload_pos = states[:, 0:3]
    payload_vel = states[:, 3:6]
    cable_direction = []
    cable_ang_vel = []
    robot_pos = []
    robot_vel = []
    robot_rot = []
    robot_body_ang_vel = []

    for robot in range(n_robots):
        directions = states[:,6 + 13*robot:9 + 13*robot]
        cable_direction.append(directions)
        angular_vel = states[:,9 + 13*robot:12 + 13*robot]
        cable_ang_vel.append(angular_vel)
        positon = payload_pos - cable_lengths[robot]*directions
        robot_pos.append(positon)
        velocity = payload_vel - cable_lengths[robot]*np.cross(angular_vel, directions)
        robot_vel.append(velocity)

        robot_rot.append(states[:,12 + 13*robot:16 + 13*robot])
        robot_body_ang_vel.append(states[:,16 + 13*robot:19 + 13*robot])

    data = {
        'start': {
            'payload': payload_pos[0].tolist(),
        },
        'payload_pos': payload_pos.tolist(),
    }

    for robot in range(n_robots):
        data['start'].update({f'robot_{robot}': robot_pos[robot][0].tolist()})
        data['start'].update({f'cable_{robot}_direction': cable_direction[robot][0].tolist()})
        data.update({f'robot_{robot}_pos': robot_pos[robot].tolist()})
        data.update({f'cable_{robot}_direction': cable_direction[robot].tolist()})
    with open('start_pos.yaml', 'w') as file:
        yaml.dump(data, file, sort_keys=False, default_flow_style=False)


