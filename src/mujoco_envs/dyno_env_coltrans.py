from typing import Optional

import dynobench
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from src.util.load_traj import load_coltans_traj


class DynoColtransEnv(gym.Env):
    def __init__(
            self,
            model,
            model_path,
            reference_traj_path,
            num_robots
    ):
        super().__init__()

        self.num_robots = num_robots
        self.action_space = Box(low=0, high=0.14, shape=(4 * self.num_robots,), dtype=np.float64)
        observation_space_size = (6
                                  + 6 * self.num_robots
                                  + 7 * self.num_robots
                                  )

        self.observation_space = Box(low=-np.inf, high=np.inf,
                                shape=(observation_space_size,), dtype=np.float64)
        self.robot = dynobench.robot_factory(
            model_path, [-1000, -1000, -1.0], [1000, 1000, 1.0]
        )
        self.steps = 0
        self.states_d, self.actions_d = load_coltans_traj(reference_traj_path)
        self.initState = self.states_d[0]
        self.state = self.initState
        self.payloadStSize = 6
        print(self.num_robots)
        print(self.state)
        self.states = np.zeros(
            (len(self.states_d), self.payloadStSize + 6 * self.num_robots + 7 * self.num_robots)
        )
        self.states[0] = self.initState
        self.appSt = []
        self.appU = []
        self.u = 0

        self.dt = model["dt"]

    def step(self, action):

        payload_pos = self.state[0:3]

        xnext = self.states[self.steps + 1]
        x = self.states[self.steps]

        self.robot.step(xnext, x, action, self.dt)
        self.steps += 1

        self.state = xnext
        self.u = action
        self.appSt.append(self.state.tolist())
        self.appU.append(self.u.tolist())

        observation = self._get_obs()
        reward = self._get_reward(action, payload_pos)
        done, distance_truncated, time_limited_truncated = self._get_done_or_truncated(payload_pos)
        info = self._get_info(reward, done, distance_truncated, time_limited_truncated)

        return observation, reward, done, distance_truncated or time_limited_truncated, info

    def _get_obs(self):
        return self.state

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.state = self.initState
        self.payloadStSize = 6
        self.states = np.zeros(
            (len(self.states_d), self.payloadStSize + 6 * self.num_robots + 7 * self.num_robots)
        )
        self.states[0] = self.initState
        self.appSt = []
        self.appU = []
        self.u = 0
        self.steps = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _next_d_state_and_action(self):
        return self.states_d[self.steps], self.actions_d[self.steps]

    def _get_info(self, reward=0, done=False, distance_truncated=False, time_limited_truncated=False):
        payload_pos_error = np.linalg.norm(self.states_d[self.steps, 0:3] - self.state[0:3])
        info = {
            "reward": reward,
            "payload_pos_error": payload_pos_error
        }
        if done:
            info["done"] = True
            info["terminal_observation"] = self.state
        if distance_truncated:
            info["distance_truncated"] = True
        if time_limited_truncated:
            info["TimeLimit.truncated"] = True

        return info

    def _get_done_or_truncated(self, payload_pos):
        done = False
        distance_truncated = False
        time_limited_truncated = False
        final_payload_pos = self.states_d[-1, 0:3]
        payload_pos = self.state[0:3]
        payload_distance = np.linalg.norm(final_payload_pos - payload_pos)
        payload_pos_error = np.linalg.norm(self.states_d[self.steps, 0:3] - self.state[0:3])
        if self.steps >= len(self.actions_d):
            if payload_distance < 0.01:
                done = True
            else:
                time_limited_truncated = True

        if payload_pos_error > 0.1:
            done = True

        return done, distance_truncated, time_limited_truncated

    def _get_reward(self, action, payload_pos_before):
        payload_pos_after = self.state[0:3]
        final_payload_pos = self.states_d[-1, 0:3]

        distance_before = np.linalg.norm(final_payload_pos - payload_pos_before)
        distance_after = np.linalg.norm(final_payload_pos - payload_pos_after)

        ctrl_cost = self._control_cost(action)

        reward = - ctrl_cost

        if distance_after < 0.01:
            reward += 5.0

        return reward

    def _control_cost(self, action):
        ctrl_cost_weight = 0.1
        control_cost = ctrl_cost_weight * np.sum(np.square(action))
        return control_cost
