import os

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.evaluation import evaluate_policy

from src.dagger.policies import PIDPolicy

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer, DAggerTrainer, reconstruct_trainer
from imitation.util.util import make_vec_env
from imitation.data import rollout, serialize, types

dirname = os.path.dirname(__file__)
training_dir = dirname + "/../../training/dagger"
traj_dir = dirname + "/../../trajectories/target_trajectories/"
circle_traj_file = traj_dir + "circle0.csv"
figure8_traj_file = traj_dir + "figure8_0.csv"
helix0_traj_file = traj_dir + "helix0.csv"
lissajous0_traj_file = traj_dir + "lissajous0.csv"
oscillation_traj_file = traj_dir + "radial_oscillation0.csv"
wave_traj_file = traj_dir + "wave0.csv"

rng = np.random.default_rng(0)
device = torch.device('cpu')
beta = 0.2




def validate_policy(traj_file, algo='dagger'):

    gym.envs.registration.register(
        id='PointMass-validate',
        entry_point='mujoco_env_pid:PointMassEnv',
        kwargs={
            'dagger': algo,
            'traj_file': traj_file
        },
    )
    env_id = "PointMass-validate"
    pm_venv = make_vec_env(
        env_id,
        rng=rng,
        n_envs=1,
        parallel=False
    )


    policy = bc.reconstruct_policy(policy_path=training_dir+"/policy-latest.pt",
                                   device=device,)
    rollout_round_min_timesteps = 200


    sample_until = rollout.make_sample_until(
        min_timesteps=rollout_round_min_timesteps,
        min_episodes=1,
    )

    trajectories = rollout.generate_trajectories(
        policy=policy,
        venv=pm_venv,
        sample_until=sample_until,
        deterministic_policy=True,
        rng=rng,
    )

    print("done")