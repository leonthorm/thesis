# Script for running ThriftyDAgger
from src.thrifty.algos.thriftydagger import thrifty
import torch
import gymnasium as gym

import numpy as np

from imitation.util.util import make_vec_env
from src.dagger.pid_policy import PIDPolicy

rng = np.random.default_rng(0)
device = torch.device('cpu')

if __name__ == '__main__':


    gym.envs.registration.register(
        id='PointMass-v1',
        entry_point='mujoco_env_pid:PointMassEnv',
        # kwargs={'target_state': target_state},
    )

    beta = 0.2

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


    thrifty(pm_venv,
            iters=20,
            expert_policy=expert,
            input_file="input_data.pkl",
            num_nets=5)