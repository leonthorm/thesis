import re
from typing import Optional

import dynobench
import gymnasium as gym
import numpy as np
import yaml
from gymnasium.spaces import Box

from src.util.helper import calculate_observation_space_size, derivative
from src.util.load_traj import load_coltans_traj


class DynoColtransEnv(gym.Env):
    def __init__(
            self,
            model,
            model_path,
            reference_traj_path,
            num_robots,
            validate: bool = False,
            validate_out: str = None
    ):

        self.dt = model["dt"]
        self.num_robots = num_robots
        self.action_space = Box(low=0, high=1.5, shape=(4 * self.num_robots,), dtype=np.float64)

        self.observation_space = Box(low=-np.inf, high=np.inf,
                                     shape=(calculate_observation_space_size(self.num_robots),), dtype=np.float64)
        self.robot = dynobench.robot_factory(
            model_path, [-1000, -1000, -1.0], [1000, 1000, 1.0]
        )

        self.refresult = load_coltans_traj(reference_traj_path)
        self.reference_traj_name = self._get_reference_traj_name(reference_traj_path)

        self.states_d =  np.array(self.refresult['refstates'])
        self.actions_d =  np.array(self.refresult['actions_d'])
        self.steps = 0
        self.max_steps = len(self.actions_d)
        v = np.array(self.states_d[:, 3: 6])

        self.acc_d = derivative(v, self.dt)
        self.initState = np.delete(self.states_d[0], [6, 7, 8])
        self.state = self.initState
        self.payloadStSize = 6
        self.states = np.zeros(
            (len(self.states_d), self.payloadStSize + 6 * self.num_robots + 7 * self.num_robots)
        )
        self.states[0] = self.initState
        self.appSt = []
        self.appU = []
        self.u = np.zeros(4 * self.num_robots)

        self.safe_expert_rollout = True
        self.validate = validate
        self.validate_out = validate_out

        super().__init__()

    def step(self, action):

        payload_pos = self.state[0:3]
        payload_vel = self.state[3:6]

        xnext = self.states[self.steps + 1]
        x = self.states[self.steps]

        self.robot.step(xnext, x, action, self.dt)
        self.steps += 1
        # print(self.steps)

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


        if (self.steps < self.max_steps):
            action_d = self.actions_d[self.steps]
        else:
            action_d = np.zeros(self.num_robots * 4)
        obs = np.concatenate((self.state, self.states_d[self.steps], self.acc_d[self.steps], action_d))
        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if self.safe_expert_rollout and self.steps >= 5:
            self.safe_rollout_to_yaml()

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
        self.u = np.zeros(4 * self.num_robots)
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
        if self.steps >= self.max_steps:
            if payload_distance < 0.1:
                done = True
            else:
                time_limited_truncated = True

        if payload_pos_error > 0.5:
            done = True

        if self.validate and (done or distance_truncated or time_limited_truncated):
            self.safe_rollout_to_yaml()

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

    def safe_rollout_to_yaml(self):
        self.safe_expert_rollout = False

        output = {}
        output["feasible"] =  1
        output["cost"] =  5.07
        output["result"] = {}
        output["result"]["states"] = self.appSt
        output["result"]["refstates"] = self.states_d.tolist()
        output["result"]["actions"] = self.appU
        output["result"]["actions_d"] = self.actions_d.tolist()
        output["result"]["accelerations"] = self.acc_d.tolist()
        # if args.write:
        print("Writing")
        # out = args.out
        out = f"results/dagger/expert_{self.reference_traj_name}.yaml"
        if self.validate:
            out = self.validate_out
        with open(out, "w") as file:
            yaml.safe_dump(output, file, default_flow_style=None)

    def set_reference_traj(self, reference_traj_path):
        self.reference_traj_name = self._get_reference_traj_name(reference_traj_path)
        print('reference trajectory: ', self.reference_traj_name)
        self.refresult = load_coltans_traj(reference_traj_path)
        self.states_d = np.array(self.refresult['refstates'])
        self.actions_d = np.array(self.refresult['actions_d'])
        self.steps = 0
        self.max_steps = len(self.actions_d)
        v = np.array(self.states_d[:, 3: 6])

        self.acc_d = derivative(v, self.dt)
        self.initState = np.delete(self.states_d[0], [6, 7, 8])
        self.reset()

    def _get_reference_traj_name(self, reference_traj_path):
        match = re.search(r'/([^/]+)\.yaml$', reference_traj_path)
        reference_traj_name  = match.group(1)

        return reference_traj_name
