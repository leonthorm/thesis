import argparse
import os
import subprocess

import gymnasium as gym
import numpy as np
import torch as th

from imitation.algorithms import bc
from imitation.util.util import make_vec_env
from imitation.data import rollout, rollout_multi_robot

from scripts.analysis.visualize_payload import quad3dpayload_meshcatViewer
from src.thrifty.algos.thriftydagger_venv import test_agent
from src.util.load_traj import load_model, load_coltans_traj

rng = np.random.default_rng(0)
device = th.device('cpu')

def run_visualizer(filename_env, filename_result, filename_output):
    quad3dpayload_meshcatViewer(filename_env, filename_result, filename_output, robot='point')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--inp",
        default=None,
        type=str,
        help="yaml input reference trajectory",
        required=True,
    )
    parser.add_argument(
        "-re",
        "--ref_environment",
        type=str,
        help="input reference trajectory environment",
        default=None,
    )
    parser.add_argument(
        "--out",
        default=None,
        type=str,
        help="yaml output trajectory by policy",
        required=True,
    )
    parser.add_argument(
        "-vo",
        "--vis_out",
        default=None,
        type=str,
        help="visualization output html",
        required=True,
    )
    parser.add_argument(
        "-m","--model_path", default=None, type=str, required=True, help="number of robots"
    )
    parser.add_argument(
        "-p", "--policy_path", default=None, type=str, required=True, help="policy to validate"
    )
    # parser.add_argument(
    #     "-cff", "--enable_cffirmware", action="store_true"
    # )  # on/off flag    args = parser.args
    parser.add_argument(
        "-w", "--write", action="store_true"
    )  # on/off flag    args = parser.args

    parser.add_argument(
        "-alg", "--daggerAlgorithm", default="dagger", type=str, required=False, help="dagger or thrifty"
    )
    parser.add_argument(
        "-dc", "--decentralizedPolicy", action="store_true"
    )

    args = parser.parse_args()

    model, num_robots = load_model(args.model_path)
    ref_environment = args.ref_environment
    algorithm = args.daggerAlgorithm
    decentralized = args.decentralizedPolicy

    output_file = args.out
    refresult = load_coltans_traj(args.inp)
    states_d = np.array(refresult['refstates'])
    actions_d = np.array(refresult['actions_d'])

    gym.envs.registration.register(
        id='dyno_coltrans-validate',
        entry_point='src.mujoco_envs.dyno_env_coltrans:DynoColtransEnv',
        kwargs={
            'model': model,
            'model_path': args.model_path,
            'reference_traj_path': args.inp,
            'num_robots': num_robots,
            'algorithm': algorithm,
            'validate': True,
            'validate_out': output_file,
        },
    )

    env_id = "dyno_coltrans-validate"
    venv = make_vec_env(
        env_id,
        rng=rng,
        n_envs=1,
        parallel=False
    )


    # todo: check
    rollout_round_min_episodes = 2
    rollout_round_min_timesteps = len(states_d)

    sample_until = rollout.make_sample_until(
        min_timesteps=rollout_round_min_timesteps,
        min_episodes=rollout_round_min_episodes,
    )

    if algorithm == 'dagger':
        policy = bc.reconstruct_policy(policy_path=args.policy_path, device=device)
        if decentralized:
            trajectories = rollout_multi_robot.generate_trajectories_multi_robot(
                policy=policy,
                venv=venv,
                sample_until=sample_until,
                deterministic_policy=True,
                rng=rng,
                num_robots=num_robots
            )
        else:
            trajectories = rollout.generate_trajectories(
                policy=policy,
                venv=venv,
                sample_until=sample_until,
                deterministic_policy=True,
                rng=rng,
            )
    elif algorithm == 'thrifty':
        policy = th.load(args.policy_path, map_location=device, weights_only=False).to(device)
        policy.device = device
        obs_dim = venv.observation_space.shape
        act_dim = venv.action_space.shape[0]
        act_limit = venv.action_space.high[0]
        if decentralized:
            raise NotImplementedError
            # trainer = thrifty_multi_robot(
            #     venv=venv,
            #     iters=iters,
            #     scratch_dir=str(training_dir_thrifty),
            #     device=torch.device('cpu'),
            #     observation_space=observation_space,
            #     action_space=action_space,
            #     rng=rng,
            #     expert_policy='ColtransPolicy',
            #     total_timesteps=total_timesteps,
            #     rollout_round_min_episodes=rollout_round_min_episodes,
            #     rollout_round_min_timesteps=rollout_round_min_timesteps,
            #     n_robots=num_robots,
            # )
            # logger.info("Training with decentralized Thrifty...")
            # trainer_save_path = trainer.save_trainer()
            # logger.info("Trainer saved at: %s", trainer_save_path)

        else:
            # trajectories = rollout.generate_trajectories(
            #     policy=policy,
            #     venv=venv,
            #     sample_until=sample_until,
            #     deterministic_policy=False,
            #     rng=rng,
            # )
            test_ret = test_agent(venv, policy, act_dim, act_limit, num_test_episodes=1)
            print(np.sum(test_ret["rew"]))
            print(test_ret["act"])


    run_visualizer(ref_environment, output_file, args.vis_out)



    # print(trajectories)
    print(states_d[-1])
