import numpy as np
from imitation.algorithms.dagger import DAggerTrainer
from imitation.data import rollout
from imitation.algorithms import bc
from src.dagger.policies import PIDPolicy, DbCbsPIDPolicy


def thrifty(venv, iters, scratch_dir, device, observation_space, action_space, rng, expert_policy, total_timesteps,
            rollout_round_min_episodes,
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

    # initial traj collection
    switch2robot_thresh, switch2human_thresh, switch2human_thresh2, switch2robot_thresh2 = train_on_expert_rollout(
        dagger_trainer, expert, bc_trainer, rollout_round_min_timesteps=200, rollout_episodes=3)

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


def train_on_expert_rollout(dagger_trainer, expert, bc_trainer, rollout_round_min_timesteps, rollout_episodes):
    collector = dagger_trainer.create_thrifty_trajectory_collector([], [], [], [], is_initial_collection = True)
    is_initial_collection = True

    sample_until = rollout.make_sample_until(
        min_timesteps=rollout_round_min_timesteps,
        min_episodes=rollout_episodes,
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
