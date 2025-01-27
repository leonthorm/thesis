import numpy as np
from imitation.algorithms.dagger import DAggerTrainer
from imitation.data import rollout, rollout_multi_robot
from imitation.algorithms import bc_multi_robot, bc
from imitation.policies import base as policy_base
from stable_baselines3.common import policies, torch_layers
from src.dagger.policies import PIDPolicy, DbCbsPIDPolicy
from gymnasium.spaces import Box
import torch as th
import gymnasium as gym

from imitation.algorithms.dagger_multi_robot import DAggerTrainer2Robot


def dagger(venv, iters, scratch_dir, device, observation_space, action_space, rng, expert_policy, total_timesteps, rollout_round_min_episodes,
           rollout_round_min_timesteps):

    if expert_policy == 'DbCbsPIDPolicy':
        expert = DbCbsPIDPolicy(
            observation_space=observation_space,
            action_space=action_space
        )
    else:
        expert = PIDPolicy(
            observation_space=observation_space,
            action_space=action_space
        )

    bc_trainer = bc.BC(
        observation_space=observation_space,
        action_space=action_space,
        rng=rng,
        device=device,
    )

    dagger_trainer = DAggerTrainer(
        venv=venv,
        scratch_dir=scratch_dir,
        bc_trainer=bc_trainer,
        rng=rng,
    )

    total_timestep_count = 0
    round_num = 0

    for t in range(iters):
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
        # data = bc_trainer.get_dataset()

        # collector.estimate_switch_parameters(data)
        round_num += 1

    return dagger_trainer


def dagger_2_robot(venv, iters, scratch_dir, device, observation_space, action_space, rng, expert_policy, total_timesteps, rollout_round_min_episodes,
           rollout_round_min_timesteps):

    observation_space = Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64)
    action_space = Box(low=-5.0, high=5.0, shape=(3,), dtype=np.float64)
    if expert_policy == 'DbCbsPIDPolicy':
        expert = DbCbsPIDPolicy(
            observation_space=observation_space,
            action_space=action_space
        )
    else:
        expert = PIDPolicy(
            observation_space=observation_space,
            action_space=action_space
        )

    extractor = (
        torch_layers.CombinedExtractor
        if isinstance(observation_space, gym.spaces.Dict)
        else torch_layers.FlattenExtractor
    )
    policy = policies.ActorCriticPolicy(
        observation_space=observation_space,
        action_space=action_space,
        # Set lr_schedule to max value to force error if policy.optimizer
        # is used by mistake (should use self.optimizer instead).
        lr_schedule=lambda _: th.finfo(th.float32).max,
        features_extractor_class=extractor,
        net_arch=[64, 64, 64]
    )

    bc_trainer = bc_multi_robot.BC(
        observation_space=observation_space,
        action_space=action_space,
        rng=rng,
        device=device,
        policy=policy,
    )

    dagger_trainer = DAggerTrainer2Robot(
        venv=venv,
        scratch_dir=scratch_dir,
        bc_trainer=bc_trainer,
        rng=rng,
    )

    total_timestep_count = 0
    round_num = 0

    for t in range(iters):
        print(f"Starting round {t}")
        round_episode_count = 0
        round_timestep_count = 0

        collector = dagger_trainer.create_trajectory_collector()

        sample_until = rollout.make_sample_until(
            min_timesteps=max(rollout_round_min_timesteps, dagger_trainer.batch_size),
            min_episodes=rollout_round_min_episodes,
        )

        trajectories = rollout_multi_robot.generate_trajectories_2_robots(
            policy=expert,
            venv=collector,
            sample_until=sample_until,
            deterministic_policy=True,
            rng=collector.rng,
        )
        #
        # for traj in trajectories:
        #     dagger_trainer._logger.record_mean(
        #         "dagger/mean_episode_reward",
        #         np.sum(traj.rews),
        #     )
        #     round_timestep_count += len(traj)
        #     total_timestep_count += len(traj)
        #
        # round_episode_count += len(trajectories)
        #
        # dagger_trainer._logger.record("dagger/total_timesteps", total_timestep_count)
        # dagger_trainer._logger.record("dagger/round_num", round_num)
        # dagger_trainer._logger.record("dagger/round_episode_count", round_episode_count)
        # dagger_trainer._logger.record("dagger/round_timestep_count", round_timestep_count)
        # print(round_timestep_count)
        dagger_trainer.extend_and_update()
        # data = bc_trainer.get_dataset()

        # collector.estimate_switch_parameters(data)
        round_num += 1

    return dagger_trainer
