# Script for running ThriftyDAgger
import os

from src.thrifty.algos.thriftydagger import thrifty
import torch
import gymnasium as gym
import numpy as np
import wandb

from imitation.util.util import make_vec_env
from src.dagger.policies import PIDPolicy

rng = np.random.default_rng(0)
device = torch.device('cpu')

dirname = os.path.dirname(__file__)
traj_file = dirname+"/../../../trajectories/target_trajectories/circle0.csv"
expert_data_file = dirname+"/../../../trajectories/target_trajectories/expert_data.pkl"

if __name__ == '__main__':

    # Initialize WandB
    wandb.init(
        project="thrifty-dagger",  # Replace with your WandB project name
        name="pointmass_thriftydagger_run",  # Custom name for the run
        config={
            "env_id": "PointMass-v1",
            "algorithm": "ThriftyDAgger",
            "num_nets": 5,
            "iterations": 20,
        }
    )

    # Register the custom environment
    gym.envs.registration.register(
        id='PointMass-v1',
        entry_point='src.dagger.mujoco_env_pid:PointMassEnv',
        kwargs={
            'dagger': 'thrifty',
            'traj_file': traj_file
        },
    )

    env_id = "PointMass-v1"
    pm_venv = make_vec_env(
        env_id,
        rng=rng,
        n_envs=1,
        parallel=False
    )

    expert = PIDPolicy(
        observation_space=pm_venv.observation_space,
        action_space=pm_venv.action_space
    )

    # ThriftyDAgger training
    input_file = "trajectories/dagger/expert_data.pkl"
    num_nets = 5
    iterations = 20

    thrifty(pm_venv,
            iters=20,
            expert_policy=expert,
            input_file=expert_data_file,
            num_nets=5)

    # wandb.log({
    #     "iteration": iter_num,
    #     "average_reward": avg_reward
    # })
    # print(f"Iteration {iter_num}, Average Reward: {avg_reward}")


    wandb.finish()
