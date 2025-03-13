import wandb

import os
import shutil

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.evaluation import evaluate_policy

from src.policies.policies import PIDPolicy
from scripts.validate_policy_old import validate_policy
from scripts.analysis.analysis import get_metrics
from imitation.algorithms import bc
from imitation.algorithms.dagger import DAggerTrainer
from imitation.util.util import make_vec_env
from imitation.data import rollout

dirname = os.path.dirname(__file__)
training_dir = dirname + "/../../training/dagger"
traj_dir = dirname + "/../../trajectories/expert_trajectories/"
circle_traj_file = traj_dir + "circle0.csv"
figure8_traj_file = traj_dir + "figure8_0.csv"
helix0_traj_file = traj_dir + "helix0.csv"
lissajous0_traj_file = traj_dir + "lissajous0.csv"
oscillation_traj_file = traj_dir + "radial_oscillation0.csv"
wave_traj_file = traj_dir + "wave0.csv"

rng = np.random.default_rng(0)
device = torch.device('cpu')

gym.envs.registration.register(
    id='PointMass-v0',
    entry_point='mujoco_env_pid:PointMassEnv',
    kwargs={
        'dagger': 'dagger',
        'traj_file': circle_traj_file,
        # 'render_mode': 'human'
        'render_mode': 'rgb_array'
    },
)


def train(venv, scratch_dir, bc_trainer, rng, total_timesteps, rollout_round_min_episodes, rollout_round_min_timesteps):
    dagger_trainer = DAggerTrainer(
        venv=venv,
        scratch_dir=scratch_dir,
        bc_trainer=bc_trainer,
        rng=rng,
    )

    total_timestep_count = 0
    round_num = 0
    while total_timestep_count < total_timesteps:
        print(f"Starting round {total_timestep_count}")
        collector = dagger_trainer.create_trajectory_collector()
        round_episode_count = 0
        round_timestep_count = 0

        collector = dagger_trainer.create_trajectory_collector()

        sample_until = rollout.make_sample_until(
            min_timesteps=max(rollout_round_min_timesteps, dagger_trainer.batch_size),
            min_episodes=rollout_round_min_episodes,
        )

        trajectories = rollout.generate_trajectories(
            policy=expert,
            venv=collector,
            sample_until=sample_until,
            deterministic_policy=True,
            rng=collector.rng,
        )

        for traj in trajectories:
            dagger_trainer._logger.record_mean(
                "dagger/mean_episode_reward",
                np.sum(traj.rews),
            )
            round_timestep_count += len(traj)
            total_timestep_count += len(traj)

        round_episode_count += len(trajectories)

        dagger_trainer._logger.record("dagger/total_timesteps", total_timestep_count)
        dagger_trainer._logger.record("dagger/round_num", round_num)
        dagger_trainer._logger.record("dagger/round_episode_count", round_episode_count)
        dagger_trainer._logger.record("dagger/round_timestep_count", round_timestep_count)
        print(round_timestep_count)
        dagger_trainer.extend_and_update()
        expert.reset_controller()
        round_num += 1

    print(dagger_trainer.save_trainer())
    return evaluate_policy(dagger_trainer.policy, pm_venv, 10)


def create_venv(env_id):
    venv = make_vec_env(
        env_id,
        rng=rng,
        n_envs=5,
        parallel=False
    )
    trajs = [lissajous0_traj_file,
             oscillation_traj_file,
             circle_traj_file,
             helix0_traj_file,
             figure8_traj_file,

             wave_traj_file
             ]
    for idx, env in enumerate(venv.envs):
        attr = env.get_wrapper_attr('set_traj')
        attr(trajs[idx])

    return venv


if __name__ == '__main__':

    demo_dir = os.path.abspath(training_dir + "/demos")

    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)

    wandb.init(
        project="dagger-training",  # Replace with your WandB project name
        name="pointmass_dagger_run",  # Custom name for the run
        config={
            "rollout_round_min_episodes": 3,
            "rollout_round_min_timesteps": 200,
            "total_timesteps": 4_000,
        }
    )

    env_id = "PointMass-v0"
    pm_venv = create_venv(env_id)

    expert = PIDPolicy(
        observation_space=pm_venv.observation_space,
        action_space=pm_venv.action_space,
    )

    bc_trainer = bc.BC(
        observation_space=pm_venv.observation_space,
        action_space=pm_venv.action_space,
        rng=rng,
        device=device,
    )

    reward, _ = train(venv=pm_venv, scratch_dir=training_dir, bc_trainer=bc_trainer, rng=rng,
                      total_timesteps=wandb.config.total_timesteps,
                      rollout_round_min_episodes=wandb.config.rollout_round_min_episodes,
                      rollout_round_min_timesteps=wandb.config.rollout_round_min_timesteps,
                      )

    validate_policy(wave_traj_file, 'dagger')
    trajectory = np.loadtxt(traj_dir+"../dagger/trajectory_dagger_wave.csv", delimiter=",")
    state_error_mean, state_error_std, vel_error_mean, vel_error_std = get_metrics(False, trajectory)
    wandb.log({
        "mean state error": state_error_mean,
        "std state error ": vel_error_mean,
        "mean velocity error": state_error_mean,
        "std velocity error": vel_error_std
    })
    wandb.log({"final_reward": reward})

    wandb.finish()
