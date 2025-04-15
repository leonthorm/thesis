# Script for running ThriftyDAgger
import os

from src.thrifty_og.algos.thriftydagger import thrifty
import torch
import gymnasium as gym
import src.thrifty_og.algos.core as core
import torch.nn as nn

import numpy as np

from src.policies.policies import PIDPolicy

rng = np.random.default_rng(0)
device = torch.device('cpu')

dirname = os.path.dirname(__file__)
traj_file = dirname+"/../../../trajectories/expert_trajectories/circle0.csv"
input_file = dirname+"/../../../trajectories/dagger/trajectory_expert_circle.pkl"

if __name__ == '__main__':


    gym.envs.registration.register(
        id='PointMass-v1',
        entry_point='src.dagger.mujoco_env_pid:PointMassEnv',
        kwargs={
            'dagger': 'thrifty_og',
            'traj_file': traj_file
        },
    )


    env_id = "PointMass-v1"
    env = gym.make(env_id)
    # pm_venv = make_vec_env(
    #     env_id,
    #     rng=rng,
    #     n_envs=1,
    #     parallel=False
    # )

    expert = PIDPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space
    )
    #policy
    policy = core.Ensemble
    ac_kwargs = dict(hidden_sizes=(256, 256),activation=nn.ReLU)
    num_nets = 5
    # policy training
    grad_steps = 500
    pi_lr = 1e-3
    bc_epochs = 5
    batch_size = 100

    # algorithm
    obs_per_iter = 700
    iters = 10

    #q learning
    q_learning = True
    num_test_episodes=10
    gamma = 0.9999




    thrifty(env,
            iters=iters,
            actor_critic=policy,
            ac_kwargs=ac_kwargs,
            grad_steps=grad_steps,
            obs_per_iter=obs_per_iter,
            pi_lr=pi_lr,
            batch_size=batch_size,
            bc_epochs=bc_epochs,
            q_learning=q_learning,
            num_test_episodes=num_test_episodes,
            gamma=gamma,
            expert_policy=expert,
            input_file=input_file,
            num_nets=5)