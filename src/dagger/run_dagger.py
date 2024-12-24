import os
import tempfile
import shutil

import gymnasium as gym
import numpy as np
import torch
from pygame.draw import circle
from stable_baselines3.common.evaluation import evaluate_policy

from src.dagger.pid_policy import PIDPolicy

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer, DAggerTrainer
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

#logging.getLogger().setLevel(logging.INFO)

#target_state = np.concatenate([np.random.uniform(0, 0.5, 3), [0.0, 0.0, 0.0]]).flatten()
# target_state = np.array([0.5,0.25,0.5, 0, 0, 0])

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

beta = 0.2


if __name__ == '__main__':

    demo_dir = os.path.abspath(training_dir + "/demos")

    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)


    env_id = "PointMass-v0"
    pm_venv = make_vec_env(
        env_id,
        rng=rng,
        n_envs=1,
        parallel=False
    )

    trajs = [circle_traj_file,
             lissajous0_traj_file,
             oscillation_traj_file,

             helix0_traj_file,
             figure8_traj_file,

             wave_traj_file
             ]
    for idx, env in enumerate(pm_venv.envs):
        attr = env.get_wrapper_attr('set_traj')
        attr(trajs[idx])


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


    dagger_trainer = DAggerTrainer(
        venv=pm_venv,
        scratch_dir=training_dir,
        bc_trainer=bc_trainer,
        rng=rng,
    )
    rollout_round_min_episodes = 3
    rollout_round_min_timesteps = 200

    total_timesteps = 4_000
    total_timestep_count = 0
    round_num = 0

    while total_timestep_count < total_timesteps:
        print(f"Starting round {total_timestep_count}")
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

    reward, _ = evaluate_policy(dagger_trainer.policy, pm_venv, 10)

    print(dagger_trainer.save_trainer())
    # print(bc_trainer.save)


    shutil.rmtree(training_dir + "/demos")
    print("Reward:", reward)




