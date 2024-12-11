import tempfile

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.evaluation import evaluate_policy

from src.dagger.pid_policy import PIDPolicy

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.util.util import make_vec_env


rng = np.random.default_rng(0)
device = torch.device('cpu')
beta = 0.2

#logging.getLogger().setLevel(logging.INFO)

#target_state = np.concatenate([np.random.uniform(0, 0.5, 3), [0.0, 0.0, 0.0]]).flatten()
# target_state = np.array([0.5,0.25,0.5, 0, 0, 0])

target_traj = "trajectories/target_trajectories/circle0.csv"

gym.envs.registration.register(
    id='PointMass-v0',
    entry_point='mujoco_env_pid:PointMassEnv',
    kwargs={'traj_file': target_traj},
)

beta = 0.2


if __name__ == '__main__':

    env_id = "PointMass-v0"
    pm_venv = make_vec_env(
        env_id,
        rng=rng,
        n_envs=1,
        parallel=False
    )


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

    training_dir = "../../mujoco_test/pid_controller_expert/training"
    with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
        print(tmpdir)
        dagger_trainer = SimpleDAggerTrainer(
            venv=pm_venv,
            scratch_dir=tmpdir,
            expert_policy=expert,
            bc_trainer=bc_trainer,
            rng=rng,
        )
        dagger_trainer.train(4_000)

    reward, _ = evaluate_policy(dagger_trainer.policy, pm_venv, 10)

    # print(type(dagger_trainer.policy))
    # dagger_trainer.policy.save(training_dir + "policy.pth")
    # # dagger_trainer.policy.save(training_dir + "/policies/sdt")
    # print("Reward:", reward)




    # with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
    #     print(tmpdir)
    #     dagger_trainer = DAggerTrainer(
    #         venv=pm_venv,
    #         scratch_dir=tmpdir,
    #         bc_trainer=bc_trainer,
    #         rng=rng,
    #     )
    #
    #     collector = dagger_trainer.create_trajectory_collector()
    #     obs = collector.reset()
    #
    #     max_rounds = 10
    #     for round_num in range(max_rounds):
    #         print(f"Starting round {round_num}")
    #
    #         # Collect demonstrations
    #         collector = dagger_trainer.create_trajectory_collector()
    #         obs = collector.reset()
    #         done = False
    #         while not done:
    #             expert_actions = expert.predict(obs)
    #             try:
    #                 collector.step_async(expert_actions)
    #                 obs, _, dones, _ = collector.step_wait()
    #                 done = dones.any()
    #             except Exception as e:
    #                 print(f"Error during trajectory collection: {e}")
    #                 done = True
    #
    #         # Extend and update
    #         try:
    #             dagger_trainer.extend_and_update()
    #         except Exception:
    #             print("No demonstrations available for this round. Collect more and retry.")
    #             break

    #
    #
    #
    # reward, _ = evaluate_policy(dagger_trainer.policy, pm_venv, 10)
    # print("Reward:", reward)


