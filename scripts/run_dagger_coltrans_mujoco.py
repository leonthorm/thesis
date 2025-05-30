import logging
import os
import re
import shutil

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from imitation.util.util import make_vec_env
from src.thrifty.thrifty import thrifty_multi_robot
from src.util.generate_swarm import generate_xml_from_start

from src.dagger.dagger import dagger_multi_robot
from src.util.generate_coltrans_dynamics import generate_dynamics_xml_from_start
from src.util.helper import calculate_observation_space_size_old
from src.util.load_traj import get_coltrans_state_components

dirname = os.path.dirname(__file__)
training_dir_dagger = dirname + "/../training/coltrans/dagger"
training_dir_thrifty = dirname + "/../training/coltrans/thrifty"
expert_traj_dir = dirname + "/../trajectories/expert_trajectories/coltrans_planning"
dynamics_dir = dirname + "/../src/dynamics/"

forest_4robots = expert_traj_dir + "/forest_4robots.yaml"
forest_2robots = expert_traj_dir + "/forest_2robots.yaml"
empty_1robots = expert_traj_dir + "/empty_1robots.yaml"


rng = np.random.default_rng(0)
device = torch.device('cpu')


logging.getLogger().setLevel(logging.ERROR)

# target_state = np.concatenate([np.random.uniform(0, 0.5, 3), [0.0, 0.0, 0.0]]).flatten()
# target_state = np.array([0.5,0.25,0.5, 0, 0, 0])





beta = 0.2

if __name__ == '__main__':

    expert = empty_1robots
    # text = "/forest_2robots.yaml"
    match = re.search(r'/([^/]+)\.yaml$', expert)
    expert_name = match.group(1)

    match = re.search(r'(\d+)robot', str(expert))
    num_robots = int(match.group(1))

    observation_space_size = calculate_observation_space_size_old(num_robots)
    dt = 0.01

    # algo = 'thrifty'
    algo = 'dagger'



    n_envs = 1
    cable_lengths = [0.5, 0.5, 0.5, 0.5]
    ts, payload_pos, payload_vel, cable_direction, cable_ang_vel, robot_rot, robot_pos, robot_body_ang_vel, robot_vel, actions = get_coltrans_state_components(
        expert, num_robots, dt,
        cable_lengths)
    actions_space_size = int(len(actions[0]) / num_robots)
    # todo: set quad rotation
    dynamics_xml = generate_dynamics_xml_from_start(expert_name + ".xml", num_robots, robot_pos, cable_lengths,
                                                    payload_pos, True)
    # dynamics_xml = generate_xml_from_start(expert_name + ".xml", n_robots, robot_pos, cable_lengths,
    #                                                 payload_pos, True)
    dynamics_xml = dynamics_dir + expert_name + ".xml"

    gym.envs.registration.register(
        id='coltrans-v0',
        entry_point='src.mujoco_envs.mujoco_env_coltrans:ColtransEnv',
        kwargs={
            'algo': algo,
            'traj_file': forest_2robots,
            'n_robots': num_robots,
            'observation_space_size': observation_space_size,
            'xml_file': dynamics_xml,
            'dt': dt,
            'cable_lengths': cable_lengths,
            'states_d': (
                ts,
                payload_pos, payload_vel,
                cable_direction, cable_ang_vel,
                robot_rot, robot_pos, robot_body_ang_vel, robot_vel,
                actions),
            'render_mode': 'human'
            # 'render_mode': 'rgb_array',
        },
    )
    demo_dir = os.path.abspath(training_dir_dagger + "/demos")

    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)

    env_id = "coltrans-v0"
    pm_venv = make_vec_env(
        env_id,
        rng=rng,
        n_envs=n_envs,
        parallel=False
    )

    trajs = [forest_2robots, forest_2robots, forest_2robots,
             forest_2robots]

    for idx, env in enumerate(pm_venv.envs):
        attr = env.get_wrapper_attr('set_traj')
        attr(trajs[idx])

    total_timesteps = 4_000
    rollout_round_min_episodes = 3
    rollout_round_min_timesteps = 200

    observation_space = Box(low=-np.inf, high=np.inf,
                            shape=(observation_space_size,), dtype=np.float64)
    action_space = Box(low=0, high=14, shape=(actions_space_size,), dtype=np.float64)

    if algo == 'dagger':
        dagger_trainer = dagger_multi_robot(venv=pm_venv,
                                            iters=20,
                                            scratch_dir=training_dir_dagger,
                                            device=device,
                                            observation_space=observation_space,
                                            action_space=action_space,
                                            rng=rng, expert_policy='FeedForwardPolicy', total_timesteps=total_timesteps,
                                            rollout_round_min_episodes=rollout_round_min_episodes,
                                            rollout_round_min_timesteps=rollout_round_min_timesteps,
                                            num_robots=num_robots, )
        # todo reward
        # reward, _ = evaluate_policy(dagger_trainer.policy, pm_venv, 10)
        print(dagger_trainer.save_trainer())

    if algo == 'thrifty':
        thrifty_trainer = thrifty_multi_robot(venv=pm_venv,
                                              iters=20,
                                              scratch_dir=training_dir_thrifty,
                                              device=device,
                                              observation_space=observation_space,
                                              action_space=action_space,
                                              rng=rng, expert_policy='FeedForwardPolicy',
                                              total_timesteps=total_timesteps,
                                              rollout_round_min_episodes=rollout_round_min_episodes,
                                              rollout_round_min_timesteps=rollout_round_min_timesteps,
                                              n_robots=num_robots, )

        # reward, _ = evaluate_policy(thrifty_trainer.policy, pm_venv, 10)
        print(thrifty_trainer.save_trainer())

    # if thrifty_algo and dagger_algo:
    #     evaluate_policy(thrifty_trainer.policy, pm_venv, 10)
    #     evaluate_policy(dagger_trainer.policy, pm_venv, 10)
    # print(bc_trainer.save)

    # shutil.rmtree(training_dir + "/demos")
    # print("Reward:", reward)
