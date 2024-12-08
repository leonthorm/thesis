import os
import pickle
from datetime import datetime

import numpy as np
from typing import Dict, Union
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from pid_controller import PIDController

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.1225,
    "lookat": np.array((0.0, 0.0, 0.12250000000000005)),
}
PATH = os.getcwd()

class PointMassEnv(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        # target_state: NDArray[np.float32],
        xml_file: str = PATH + "/../dynamics/point_mass.xml",
        frame_skip: int = 1,
        default_camera_config: Dict[str, Union[float, int]] = {},
        healthy_reward: float = 10.0,
        reset_noise_scale: float = 0.0,
        ctrl_cost_weight: float = 0.1,

        ** kwargs,
    ):
        observation_space = Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            render_mode="human",
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        self.target_state = np.concatenate(
            [np.random.uniform(-0.5, 0.5, 3),
            [0.0, 0.0, 0.0]], axis=0
        )
        print("target_state", self.target_state)
        # self.model.site_pos[self.model.site("target_state").id] = np.array([0, 0, 0])
        # self.model.site_pos[self.model.site_bodyid[0]] = self.target_state[0:3]
        self.pid_controller = PIDController(self.dt)
        self._ctrl_cost_weight = ctrl_cost_weight
        # for idx in range(10):
        #     x, y, z, _,_,_ = self.trajectory(idx/10)
        #     site_name = f"traj_site_{idx}"
        #     site_id = self.model.site(site_name).id
        #     self.model.site_pos[site_id] = np.array([x, y, z])
        #     print(self.model.site_pos[site_id])
        self.steps = 0
        self.max_steps = 700
        self.trajectory = np.array([])
        self.actions = np.array([])


    def step(self, action):

        position_before = self.data.qpos.copy()
        self.do_simulation(action, self.frame_skip)
        position_after = self.data.qpos.copy()

        t = self.data.time
        traj_des = self.goal_trajectory(t)

        self.target_state = traj_des
        #self.model.site_pos[self.model.site("target_state").id] = traj_des[0:3]


        observation, reward, done, info = self._get_info(action, position_after, position_before, traj_des)

        if self.render_mode == "human":
            self.render()

        self.steps += 1
        return (
            observation,
            reward,
            done,
            self.steps > self.max_steps,
            info
        )

    def _get_info(self, action, position_after, position_before, traj_des):
        distance_to_target = np.linalg.norm(self.target_state[0:3] - position_after)

        observation = self._get_obs()

        self.trajectory = np.concatenate((self.trajectory, observation))
        self.actions = np.concatenate((self.actions, action))


        reward, reward_info = self._get_rew(position_before, position_after, action)

        done = self._is_done(distance_to_target)

        info = {
            "position": self.data.qpos,
            "desired_position": traj_des[0:3],
            "velocity": self.data.qvel,
            "desired_velocity": traj_des[3:6],
            ** reward_info,
        }
        return observation, reward, done, info

    def goal_trajectory(self, t, radius=1.0, omega=2.0, z_amplitude=1.0, z_freq=0.5):

        x_d = radius * np.cos(omega * t) - radius
        y_d = radius * np.sin(omega * t)
        z_d = z_amplitude * np.sin(z_freq * t) 
        vx_d = -radius * omega * np.sin(omega * t)
        vy_d = radius * omega * np.cos(omega * t)
        vz_d = z_amplitude * z_freq * np.cos(z_freq * t)
        return np.array([x_d, y_d, 0, vx_d, vy_d, 0])


    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)
        self.steps = 0

        today = datetime.now()

        if self.trajectory.size != 0:
            print("saving trajectory")

            self.trajectory = np.reshape(self.trajectory, (-1, 12))
            self.actions = np.reshape(self.trajectory, (-1, 3))

            data = {
                'obs': self.trajectory,  # (num_samples, obs_dim)
                'act': self.actions  # (num_samples, act_dim)
            }

            np.savetxt("trajectories/trajectory_"+str(today)+".csv", self.trajectory, delimiter=",")

            pickle_filename = 'thrifty/input_data.pkl'
            with open(pickle_filename, 'wb') as f:
                pickle.dump(data, f)

        self.trajectory = np.array([])
        self.actions = np.array([])

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate(
            [self.data.qpos,
             self.data.qvel,
             self.target_state]
        ).ravel()

    def _is_done(self, distance_to_target):

        return distance_to_target > 1

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def _get_rew(self, position_before, position_after, action):

        de_weight = 10.0
        ve_weight = 0.001
        distance_before = np.linalg.norm(self.target_state[0:3] - position_before)
        distance_after = np.linalg.norm(self.target_state[0:3] - position_after)

        distance_reward = distance_before - distance_after

        velocity_current = self.data.qvel.copy()
        velocity_error = -np.linalg.norm(velocity_current)

        ctrl_cost = self.control_cost(action)

        reward = distance_reward*de_weight - ctrl_cost + velocity_error * ve_weight

        if distance_after < 0.01:
            reward += 5.0

        reward_info = {
            "distance_reward": distance_reward,
            "control_cost": -ctrl_cost,
            "velocity_error": velocity_error,
        }

        return reward, reward_info

    def get_dt(self):
        return self.dt

