import argparse
import os
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th

from imitation.algorithms import bc
from imitation.util.util import make_vec_env
from imitation.data import rollout, rollout_multi_robot

from scripts.analysis.visualize_payload import quad3dpayload_meshcatViewer
from scripts.train_dagger_coltrans_dyno import register_environment
from src.thrifty.algos.thriftydagger_venv import test_agent
from src.util.load_traj import load_model, load_coltans_traj

rng = np.random.default_rng(0)
device = th.device('cpu')


def run_visualizer(filename_env, filename_result, filename_output):
    quad3dpayload_meshcatViewer(filename_env, filename_result, filename_output, robot='point')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--inp",
        nargs="+",
        type=str,
        help="YAML input reference trajectory paths.",
        default=None,
    )
    parser.add_argument(
        "--inp_dir",
        type=str,
        help="Directory containing input reference trajectory files.",
        default=None,
    )
    parser.add_argument(
        "-re",
        "--ref_environment",
        type=str,
        help="input reference trajectory environment",
        default=None,
        required=True,
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
        "-m", "--model_path", default=None, type=str, required=True, help="number of robots"
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

    if args.inp:
        reference_paths = [Path(p) for p in args.inp]
    else:
        reference_paths = list(Path(args.inp_dir).glob("trajectory_*.yaml"))[:32]
        if not reference_paths:
            print("No trajectory files found in the specified directory: %s", args.inp_dir)
            sys.exit(1)

    model, num_robots = load_model(args.model_path)
    ref_environment = args.ref_environment
    algorithm = args.daggerAlgorithm
    decentralized = args.decentralizedPolicy

    refresult = [load_coltans_traj(ref) for ref in reference_paths]
    states_d = [np.array(ref['refstates']) for ref in refresult]

    register_environment(model, args.model_path, reference_paths[0], num_robots, algorithm)

    env_id = "dyno_coltrans-validate"
    venv = make_vec_env(
        env_id,
        rng=rng,
        n_envs=1,
        parallel=False
    )

    sample_until = rollout.make_sample_until(
        min_episodes=1,
    )

    policy = th.load(args.policy_path, map_location=device, weights_only=False).to(device)
    # policy.device = device
    deterministic_policy = True
    if algorithm == 'thrifty':
        deterministic_policy = False
    if decentralized:
        ablation_kwargs = dict(
            cable_q=True,
            cable_q_d=True,
            cable_w=True,
            cable_w_d=True,
            robot_rot=True,
            robot_rot_d=True,
            robot_w=True,
            robot_w_d=True,
            other_cable_q=True,
            other_robot_rot=True,
        )
        trajectories = rollout_multi_robot.generate_trajectories_multi_robot(
            policy=policy,
            venv=venv,
            sample_until=sample_until,
            deterministic_policy=deterministic_policy,
            rng=rng,
            num_robots=num_robots,
            **ablation_kwargs
        )
    else:
        trajectories = rollout.generate_trajectories(
            policy=policy,
            venv=venv,
            sample_until=sample_until,
            deterministic_policy=deterministic_policy,
            rng=rng,
        )
    reward = sum(
        info["reward"]
        for info in trajectories[0].infos
    )

    payload_pos_error = sum(
        info["payload_pos_error"]
        for info in trajectories[0].infos
    )

    traj_length = len(trajectories[0])


    ref_traj_length = len(states_d)-1

    traj_part_completed = traj_length / ref_traj_length
    error_per_state = payload_pos_error / traj_length
    reward_per_state = reward / traj_length
    print('######### RESULT METRICS #########')
    print(
        f'reward: {reward}, payload_pos_error: {payload_pos_error}, traj_length: {traj_length}, '
        f'ref_traj_length: {ref_traj_length}, traj completed: {traj_part_completed}, '
        f'error_per_state: {error_per_state}, reward_per_state: {reward_per_state}'
    )

    run_visualizer(ref_environment, output_file, args.vis_out)
