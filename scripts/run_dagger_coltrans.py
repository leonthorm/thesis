import os
import shutil

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from stable_baselines3.common.evaluation import evaluate_policy

from src.dagger.dagger import dagger_multi_robot

from imitation.util.util import make_vec_env

from src.thrifty.thrifty import thrifty_multi_robot
from src.util.generate_coltrans_dynamics import MuJoCoSceneGenerator
from src.util.generate_coltrans_dynamics import generate_dynamics_xml_from_start
from src.util.load_traj import load_coltans_traj, get_coltrans_state_components

dirname = os.path.dirname(__file__)
training_dir_dagger = dirname + "/../training/coltrans/dagger"
training_dir_thrifty = dirname + "/../training/coltrans/thrifty"
expert_traj_dir = dirname + "/../trajectories/expert_trajectories/coltrans_planning"
dynamics = dirname + "/../src/dynamics/"

rng = np.random.default_rng(0)
device = torch.device('cpu')


# logging.getLogger().setLevel(logging.INFO)

# target_state = np.concatenate([np.random.uniform(0, 0.5, 3), [0.0, 0.0, 0.0]]).flatten()
# target_state = np.array([0.5,0.25,0.5, 0, 0, 0])




def calculate_observation_space_size(n_robots):
    payload_pos, payload_vel = 3, 3
    cable_direction, cable_force = 3, 3
    robot_pos, robot_vel, robot_rot, robot_body_ang_vel = 3, 3, 4, 3
    other_robot_pos = 3

    size = (payload_pos + payload_vel
            + cable_direction + cable_force
            + robot_pos + robot_vel + robot_rot + robot_body_ang_vel
            + n_robots * other_robot_pos)

    return size


beta = 0.2

if __name__ == '__main__':

    n_robots = 4
    observation_space_size = calculate_observation_space_size(n_robots)
    dt = 0.01

    # algo = 'thrifty'
    algo = 'dagger'

    n_envs = 1
    cable_lengths = [0.5, 0.5, 0.5, 0.5]
    forest_4robots = expert_traj_dir + "/forest_4robots.yaml"
    ts, payload_pos, payload_vel, cable_direction, cable_ang_vel, robot_pos, robot_vel, robot_rot, robot_body_ang_vel, actions = get_coltrans_state_components(
        forest_4robots, n_robots, dt,
        cable_lengths)
    actions_space_size = int(len(actions[0]) / n_robots)
    # todo: set quad rotation
    # dynamics_xml = generate_dynamics_xml_from_start("forest_4robots.xml", n_robots, robot_pos[:, 0], cable_lengths, payload_pos[0])
    dynamics_xml = dynamics + "forest_4robots.xml"

    gym.envs.registration.register(
        id='coltrans-v0',
        entry_point='src.mujoco_envs.mujoco_env_coltrans:ColtransEnv',
        kwargs={
            'algo': algo,
            'traj_file': forest_4robots,
            'n_robots': n_robots,
            'observation_space_size': observation_space_size,
            'xml_file': dynamics_xml,
            'dt': dt,
            'cable_lengths': cable_lengths,
            'states_d': (ts, payload_pos, payload_vel, cable_direction, cable_ang_vel, robot_pos, robot_vel, robot_rot, robot_body_ang_vel, actions),
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

    trajs = [forest_4robots, forest_4robots, forest_4robots,
             forest_4robots]

    for idx, env in enumerate(pm_venv.envs):
        attr = env.get_wrapper_attr('set_traj')
        attr(trajs[idx])

    total_timesteps = 4_000
    rollout_round_min_episodes = 3
    rollout_round_min_timesteps = 200

    observation_space = Box(low=-np.inf, high=np.inf,
                            shape=(observation_space_size,), dtype=np.float64)
    action_space = Box(low=-10.0, high=10.0, shape=(actions_space_size,), dtype=np.float64)

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
                                            n_robots=n_robots, )
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
                                              n_robots=n_robots, )

        # reward, _ = evaluate_policy(thrifty_trainer.policy, pm_venv, 10)
        print(thrifty_trainer.save_trainer())

    # if thrifty_algo and dagger_algo:
    #     evaluate_policy(thrifty_trainer.policy, pm_venv, 10)
    #     evaluate_policy(dagger_trainer.policy, pm_venv, 10)
    # print(bc_trainer.save)

    # shutil.rmtree(training_dir + "/demos")
    # print("Reward:", reward)
