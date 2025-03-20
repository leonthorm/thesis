import argparse
import os
import shutil
import sys

import dynobench
import gymnasium as gym
import numpy as np
import torch
import yaml
from gymnasium.spaces import Box
from imitation.util.util import make_vec_env

from src.dagger.dagger import dagger_multi_robot, dagger
from src.thrifty.thrifty import thrifty_multi_robot, thrifty
from src.util.helper import calculate_observation_space_size
# from mujoco_test.generate_swarm import generate_xml_from_start
from src.util.load_traj import load_model

dirname = os.path.dirname(__file__)
training_dir_dagger = dirname + "/../training/coltrans/dagger"
training_dir_thrifty = dirname + "/../training/coltrans/thrifty"
expert_traj_dir = dirname + "/../trajectories/expert_trajectories/coltrans_planning"


rng = np.random.default_rng(0)
device = torch.device('cpu')

# logging.getLogger().setLevel(logging.INFO)

# target_state = np.concatenate([np.random.uniform(0, 0.5, 3), [0.0, 0.0, 0.0]]).flatten()
# target_state = np.array([0.5,0.25,0.5, 0, 0, 0])


beta = 0.2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inp",
        default=None,
        nargs="+",
        type=str,
        help="yaml input reference trajectory",
        required=True,
    )
    # parser.add_argument(
    #     "--inp_dir",
    #     default=None,
    #     type=str,
    #     help="dir with all input reference trajectory",
    #     required=False,
    # )
    parser.add_argument(
        "--env",
        default=None,
        type=str,
        help="env inflated",
        required=False,
    )
    # todo: output
    parser.add_argument(
        "--out",
        default=None,
        type=str,
        help="yaml output trajectory by policy",
        required=False,
    )
    parser.add_argument(
        "--model_path", default=None, type=str, required=True, help="number of robots"
    )
    # parser.add_argument(
    #     "-cff", "--enable_cffirmware", action="store_true"
    # )  # on/off flag    args = parser.args
    parser.add_argument(
        "-w", "--write", action="store_true"
    )  # on/off flag    args = parser.args
    parser.add_argument(
        "-a", "--compAcc", action="store_true"
    )  # on/off flag    args = parser.args
    #todo: cable tracking
    parser.add_argument(
        "-noC", "--nocableTracking", action="store_true"
    )  # on/off flag    args = parser.args

    parser.add_argument(
        "-alg", "--daggerAlgorithm", default="dagger", type=str, required=False, help="dagger or thrifty"
    )
    parser.add_argument(
        "-dc", "--decentralizedPolicy", action="store_true"
    )

    args = parser.parse_args()

    reference_paths = args.inp

    model, num_robots = load_model(args.model_path)

    with open(args.env, "r") as f:
        env = yaml.safe_load(f)
    cable_lengths = env["robots"][0]["l"]
    algorithm = args.daggerAlgorithm
    decentralized = args.decentralizedPolicy


    # robot = "robot"
    gym.envs.registration.register(
        id='dyno_coltrans-v0',
        entry_point='src.mujoco_envs.dyno_env_coltrans:DynoColtransEnv',
        kwargs={
            'model': model,
            'model_path': args.model_path,
            'reference_traj_path': reference_paths[0],
            'num_robots': num_robots,
        },
    )
    demo_dir = os.path.abspath(training_dir_dagger + "/demos")

    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)

    env_id = "dyno_coltrans-v0"
    venv = make_vec_env(
        env_id,
        rng=rng,
        n_envs=len(reference_paths),
        parallel=False
    )
    # todo: different envs

    for idx, env in enumerate(venv.envs):
        attr = env.get_wrapper_attr('set_reference_traj')
        attr(reference_paths[idx])

    # trajs = [forest_2robots, forest_2robots, forest_2robots,
    #          forest_2robots]
    #
    # for idx, env in enumerate(venv.envs):
    #     attr = env.get_wrapper_attr('set_traj')
    #     attr(trajs[idx])

    # todo: check this
    total_timesteps = 4_000
    rollout_round_min_episodes = 3
    rollout_round_min_timesteps = 400
    iters = 10

    observation_space = Box(low=-np.inf, high=np.inf,
                            shape=(calculate_observation_space_size(num_robots),), dtype=np.float64)
    action_space = Box(low=0, high=1.5, shape=(4 * num_robots,), dtype=np.float64)

    if decentralized:
        if algorithm == 'dagger':
            dagger_trainer = dagger_multi_robot(venv=venv,
                                                iters=iters,
                                                scratch_dir=training_dir_dagger,
                                                device=device,
                                                observation_space=observation_space,
                                                action_space=action_space,
                                                rng=rng, expert_policy='ColtransPolicy',
                                                total_timesteps=total_timesteps,
                                                rollout_round_min_episodes=rollout_round_min_episodes,
                                                rollout_round_min_timesteps=rollout_round_min_timesteps,
                                                num_robots=num_robots,
                                                cable_lengths=cable_lengths
                                                )
            # todo reward
            # reward, _ = evaluate_policy(dagger_trainer.policy, pm_venv, 10)
            print(dagger_trainer.save_trainer())

        if algorithm == 'thrifty':
            thrifty_trainer = thrifty_multi_robot(venv=venv,
                                                  iters=iters,
                                                  scratch_dir=training_dir_thrifty,
                                                  device=device,
                                                  observation_space=observation_space,
                                                  action_space=action_space,
                                                  rng=rng, expert_policy='ColtransPolicy',
                                                  total_timesteps=total_timesteps,
                                                  rollout_round_min_episodes=rollout_round_min_episodes,
                                                  rollout_round_min_timesteps=rollout_round_min_timesteps,
                                                  n_robots=num_robots, )

            # reward, _ = evaluate_policy(thrifty_trainer.policy, pm_venv, 10)
            print(thrifty_trainer.save_trainer())
    else:
        if algorithm == 'dagger':
            dagger_trainer = dagger(venv=venv,
                                    iters=iters,
                                    scratch_dir=training_dir_dagger,
                                    device=device,
                                    observation_space=observation_space,
                                    action_space=action_space,
                                    rng=rng, expert_policy='ColtransPolicy',
                                    total_timesteps=total_timesteps,
                                    rollout_round_min_episodes=rollout_round_min_episodes,
                                    rollout_round_min_timesteps=rollout_round_min_timesteps,
                                    num_robots=num_robots)
            # todo reward
            # reward, _ = evaluate_policy(dagger_trainer.policy, pm_venv, 10)
            print(dagger_trainer.save_trainer())

        if algorithm == 'thrifty':
            thrifty_trainer = thrifty(venv=venv,
                                      iters=iters,
                                      scratch_dir=training_dir_thrifty,
                                      device=device,
                                      observation_space=observation_space,
                                      action_space=action_space,
                                      rng=rng, expert_policy='ColtransPolicy',
                                      total_timesteps=total_timesteps,
                                      rollout_round_min_episodes=rollout_round_min_episodes,
                                      rollout_round_min_timesteps=rollout_round_min_timesteps,
                                      num_robots=num_robots, )
            print(thrifty_trainer.save_trainer())


    # if thrifty_algo and dagger_algo:
    #     evaluate_policy(thrifty_trainer.policy, pm_venv, 10)
    #     evaluate_policy(dagger_trainer.policy, pm_venv, 10)
    # print(bc_trainer.save)

    # shutil.rmtree(training_dir + "/demos")
    # print("Reward:", reward)
