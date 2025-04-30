import numpy as np
from gymnasium.spaces import Box
from torch import nn

from deps.imitation.src.imitation.algorithms.dagger import DAggerTrainer
from deps.imitation.src.imitation.data import rollout, rollout_multi_robot
from deps.imitation.src.imitation.algorithms import bc_multi_robot, bc
from stable_baselines3.common import policies, torch_layers

from deps.imitation.src.imitation.algorithms.dagger_multi_robot import DAggerTrainerMultiRobot
from deps.imitation.src.imitation.data.rollout_multi_robot import get_len_obs_single_robot
from src.policies.policies import PIDPolicy, DbCbsPIDPolicy, ColtransPolicy, ColtransPolicy1Env
import torch as th
import gymnasium as gym


def dagger(venv,
           iters,
           scratch_dir,
           device,
           observation_space,
           action_space,
           rng,
           expert_policy,
           total_timesteps,
           rollout_round_min_episodes,
           rollout_round_min_timesteps,
           num_robots=1,
           policy_kwargs=None,
           ):
    expert = get_expert(action_space, expert_policy, num_robots, observation_space, venv)

    policy = create_policy(action_space, observation_space, policy_kwargs)

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

    for t in range(iters):
        print(f"Starting round {t}")
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


def dagger_multi_robot(venv, iters, scratch_dir, device, observation_space, action_space, rng, expert_policy,
                       total_timesteps, rollout_round_min_episodes,
                       rollout_round_min_timesteps, num_robots, cable_lengths, **ablation_kwargs):
    expert = get_expert(action_space, expert_policy, num_robots, observation_space, venv, cable_lengths)

    len_obs_single_robot = get_len_obs_single_robot(num_robots, **ablation_kwargs)

    observation_space_single_robot = Box(low=-np.inf, high=np.inf,
                                         shape=(len_obs_single_robot,), dtype=np.float64)
    action_space_single_robot = Box(low=0, high=1.5, shape=(4,), dtype=np.float64)

    policy = create_policy(action_space_single_robot, observation_space_single_robot)

    bc_trainer = bc_multi_robot.BCMultiRobot(
        observation_space=observation_space,
        action_space=action_space,
        rng=rng,
        device=device,
        policy=policy,
        num_robots=num_robots,
    )

    dagger_trainer = DAggerTrainerMultiRobot(
        venv=venv,
        scratch_dir=scratch_dir,
        bc_trainer=bc_trainer,
        rng=rng,
        num_robots=num_robots,
    )

    total_timestep_count = 0
    round_num = 0

    for t in range(iters):
        print(f"Starting round {t}")
        round_episode_count = 0
        round_timestep_count = 0

        collector = dagger_trainer.create_trajectory_collector_multi_robot(
            actions_size_single_robot=4,
            num_robots=num_robots,
            cable_lengths=cable_lengths
        )

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
            num_robots=num_robots
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


def get_expert(action_space, expert_policy, num_robots, observation_space, venv, cable_lengths=[0.5, 0.5]):
    if expert_policy == 'DbCbsPIDPolicy':
        expert = DbCbsPIDPolicy(
            observation_space=observation_space,
            action_space=action_space
        )
    elif expert_policy == 'FeedForwardPolicy':
        expert = ColtransPolicy(
            observation_space=observation_space,
            action_space=action_space
        )
    elif expert_policy == 'PIDPolicy':
        expert = PIDPolicy(
            observation_space=observation_space,
            action_space=action_space
        )
    elif expert_policy == 'ColtransPolicy':
        expert = ColtransPolicy(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            num_robots=num_robots,
            n_venvs=venv.num_envs,
        )
    elif expert_policy == 'ColtransPolicy1Env':
        expert = ColtransPolicy1Env(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            num_robots=num_robots,
        )
    return expert


def create_policy(action_space, observation_space, policy_kwargs=None):
    if policy_kwargs is None:
        policy_kwargs = dict()
    if "net_arch" in policy_kwargs:
        net_arch = policy_kwargs["net_arch"]
    else:
        net_arch = [64, 64, 64]

    if "activation_fn" in policy_kwargs:
        activation_fn = policy_kwargs["activation_fn"]
    else:
        activation_fn = nn.Tanh

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
        net_arch=net_arch,
        activation_fn=activation_fn
    )
    return policy
