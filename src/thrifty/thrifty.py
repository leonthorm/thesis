import numpy as np
from imitation.algorithms.dagger import DAggerTrainer
from imitation.algorithms.dagger_multi_robot import DAggerTrainerMultiRobot
from imitation.data import rollout, rollout_multi_robot
from imitation.algorithms import bc_multi_robot, bc
from stable_baselines3.common import policies, torch_layers
import gymnasium as gym
import torch as th

from src.dagger.dagger import get_expert, get_policy
from src.policies.policies import PIDPolicy, DbCbsPIDPolicy


def thrifty(venv, iters, scratch_dir, device, observation_space, action_space, rng, expert_policy, total_timesteps,
            rollout_round_min_episodes,
            rollout_round_min_timesteps, num_robots=1):
    expert = get_expert(action_space, expert_policy, num_robots, observation_space, venv)

    policy = get_policy(action_space, observation_space)

    bc_trainer = bc.BC(
        observation_space=observation_space,
        action_space=action_space,
        rng=rng,
        device=device,
        policy=policy
    )

    dagger_trainer = DAggerTrainer(
        venv=venv,
        scratch_dir=scratch_dir,
        bc_trainer=bc_trainer,
        rng=rng,
    )

    total_timestep_count = 0
    round_num = 0

    # initial traj collection
    switch2robot_thresh, switch2human_thresh, switch2human_thresh2, switch2robot_thresh2 = train_on_expert_rollout(
        dagger_trainer, expert, bc_trainer, rollout_round_min_timesteps=200, rollout_round_min_episodes=3)

    #

    for t in range(iters):
        print(f"Starting round {total_timestep_count}")
        round_episode_count = 0
        round_timestep_count = 0
        collector = dagger_trainer.create_thrifty_trajectory_collector(switch2robot_thresh=switch2robot_thresh,
                                                                       switch2human_thresh=switch2human_thresh,
                                                                       switch2human_thresh2=switch2human_thresh2,
                                                                       switch2robot_thresh2=switch2robot_thresh2,
                                                                       is_initial_collection=False)

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

        switch2human_thresh, switch2human_thresh2, switch2robot_thresh2 = collector.recompute_thresholds()
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

        # retrain policy from scratch
        dagger_trainer = DAggerTrainer(
            venv=venv,
            scratch_dir=scratch_dir,
            bc_trainer=bc_trainer,
            rng=rng,
        )

        dagger_trainer.extend_and_update()
        round_num += 1

    return dagger_trainer


def thrifty_multi_robot(venv, iters, scratch_dir, device, observation_space, action_space, rng, expert_policy,
                        total_timesteps,
                        rollout_round_min_episodes,
                        rollout_round_min_timesteps, n_robots):
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

    bc_trainer = bc_multi_robot.BCMultiRobot(
        observation_space=observation_space,
        action_space=action_space,
        rng=rng,
        device=device,
        policy=policy,
        n_robots=n_robots,
    )

    dagger_trainer = DAggerTrainerMultiRobot(
        venv=venv,
        scratch_dir=scratch_dir,
        bc_trainer=bc_trainer,
        rng=rng,
        n_robots=n_robots,
    )

    total_timestep_count = 0
    round_num = 0

    # initial traj collection
    switch2robot_thresh, switch2human_thresh, switch2human_thresh2, switch2robot_thresh2 = train_on_expert_rollout_multi_robot(
        dagger_trainer, expert, bc_trainer, rollout_round_min_timesteps=200,
        rollout_round_min_episodes=rollout_round_min_episodes,
        actions_size_single_robot=action_space.shape[0], n_robots=n_robots, )

    #

    for t in range(iters):
        print(f"Starting round {total_timestep_count}")
        round_episode_count = 0
        round_timestep_count = 0
        collector = dagger_trainer.create_thrifty_trajectory_collector_multi_robot(
            switch2robot_thresh=switch2robot_thresh,
            switch2human_thresh=switch2human_thresh,
            switch2human_thresh2=switch2human_thresh2,
            switch2robot_thresh2=switch2robot_thresh2,
            actions_size_single_robot=action_space.shape[0],
            n_robots=n_robots,
            is_initial_collection=False)

        sample_until = rollout_multi_robot.make_sample_until(
            min_timesteps=max(rollout_round_min_timesteps, dagger_trainer.batch_size),
            min_episodes=rollout_round_min_episodes,
        )

        trajectories = rollout_multi_robot.generate_trajectories_multi_robot(
            policy=expert,
            venv=collector,
            sample_until=sample_until,
            deterministic_policy=True,
            rng=collector.rng,
            n_robots=n_robots
        )

        switch2human_thresh, switch2human_thresh2, switch2robot_thresh2 = collector.recompute_thresholds_multi_robot()
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

        # retrain policy from scratch
        dagger_trainer = DAggerTrainerMultiRobot(
            venv=venv,
            scratch_dir=scratch_dir,
            bc_trainer=bc_trainer,
            rng=rng,
            n_robots=n_robots,
        )

        dagger_trainer.extend_and_update()
        round_num += 1

    return dagger_trainer


def train_on_expert_rollout_multi_robot(dagger_trainer, expert, bc_trainer, rollout_round_min_timesteps,
                                        rollout_round_min_episodes,
                                        actions_size_single_robot, n_robots):
    collector = dagger_trainer.create_thrifty_trajectory_collector_multi_robot([], [], [], [],
                                                                               actions_size_single_robot=actions_size_single_robot,
                                                                               n_robots=n_robots,
                                                                               is_initial_collection=True)

    sample_until = rollout_multi_robot.make_sample_until(
        min_timesteps=max(rollout_round_min_timesteps, dagger_trainer.batch_size),
        min_episodes=rollout_round_min_episodes,
    )

    rollout_multi_robot.generate_trajectories_multi_robot(
        policy=expert,
        venv=collector,
        sample_until=sample_until,
        deterministic_policy=True,
        rng=collector.rng,
        n_robots=n_robots
    )
    dagger_trainer.extend_and_update()

    data = bc_trainer.get_dataset()

    switch2robot_thresh, switch2human_thresh, switch2human_thresh2, switch2robot_thresh2 = collector.estimate_switch_parameters_multi_robot(
        data)

    return switch2robot_thresh, switch2human_thresh, switch2human_thresh2, switch2robot_thresh2


def train_on_expert_rollout(dagger_trainer, expert, bc_trainer, rollout_round_min_timesteps,
                            rollout_round_min_episodes,):
    collector = dagger_trainer.create_thrifty_trajectory_collector([], [], [], [],is_initial_collection=True)

    sample_until = rollout.make_sample_until(
        min_timesteps=max(rollout_round_min_timesteps, dagger_trainer.batch_size),
        min_episodes=rollout_round_min_episodes,
    )

    rollout.generate_trajectories(
        policy=expert,
        venv=collector,
        sample_until=sample_until,
        deterministic_policy=True,
        rng=collector.rng,
    )
    dagger_trainer.extend_and_update()

    data = bc_trainer.get_dataset()

    switch2robot_thresh, switch2human_thresh, switch2human_thresh2, switch2robot_thresh2 = collector.estimate_switch_parameters(
        data)

    return switch2robot_thresh, switch2human_thresh, switch2human_thresh2, switch2robot_thresh2
