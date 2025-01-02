import os
import pickle
from datetime import datetime
import re
import numpy as np
from typing import Dict, Union
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from src.dagger.spline_traj import get_trajectory

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.1225,
    "lookat": np.array((0.0, 0.0, 0.12250000000000005)),
}
dirname = os.path.dirname(__file__)
trajectories_dir = dirname+"/../../trajectories"
class PointMassEnv(MujocoEnv):
        metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
        }
        print(trajectories_dir)
        def __init__(
            self,
            # target_state: NDArray[np.float32],
            traj_file,
            dagger,
            # xml_file: str =  "../src/dynamics/point_mass.xml",
            xml_file: str = dirname + "/../dynamics/point_mass.xml",
            frame_skip: int = 1,
            default_camera_config: Dict[str, Union[float, int]] = {},
            healthy_reward: float = 10.0,
            reset_noise_scale: float = 0.0,
            ctrl_cost_weight: float = 0.1,
            render_mode: str="rgb_array",
            ** kwargs,
        ):
            self.dagger = dagger
            observation_space = Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float64)


            MujocoEnv.__init__(
                self,
                xml_file,
                frame_skip,
                observation_space=observation_space,
                render_mode=render_mode,
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

            self.ts, self.pos_d, self.vel_d, self.acc_d, self.jerk_d, self.snap_d = get_trajectory(traj_file)
            self.target_state = np.ravel([self.pos_d[0], self.vel_d[0], self.acc_d[0]])


            self.model.site_pos[self.model.site("target_state").id] = np.array([0, 0, 0])
            # self.model.site_pos[self.model.site_bodyid[0]] = self.target_state[0:3]
            self._ctrl_cost_weight = ctrl_cost_weight
            # for idx in range(10):
            #     x, y, z, _,_,_ = self.trajectory(idx/10)
            #     site_name = f"traj_site_{idx}"
            #     site_id = self.model.site(site_name).id
            #     self.model.site_pos[site_id] = np.array([x, y, z])
            #     print(self.model.site_pos[site_id])
            self.steps = 0
            self.max_steps = 0
            self.trajectory = np.array([])
            self.observations = np.array([])
            self.actions = np.array([])
            self.max_t = 2.0
            self.first_run = True
            self.traj_name = ''
            self.set_traj_name(traj_file)
            self.current_pos_error = 0
            self.max_steps


        def set_traj(self, traj_file):
            print('target trajectory', traj_file)
            self.ts, self.pos_d, self.vel_d, self.acc_d, self.jerk_d, self.snap_d = get_trajectory(traj_file)
            self.reset_model()
            self.set_traj_name(traj_file)
            self.max_steps = len(self.ts)

        def set_traj_name(self, traj_file):
            match = re.search(r"/([^/]+?)0[^/]*$", traj_file)
            if match:
                name = match.group(1)
                self.traj_name = name

        def step(self, action):
            # print(self.data.time / self.dt)
            position_before = self.data.qpos.copy()
            self.add_state_to_trajectory(action, position_before)
            self.do_simulation(action, self.frame_skip)
            position_after = self.data.qpos.copy()
            truncation = self.set_target_state()
            # t = self.data.time
            # traj_des = self.goal_trajectory(t)

            # self.target_state = traj_des
            observation, reward, done, info = self._get_info(action, position_after, position_before)



            if self.render_mode == "human":
                self.render()

            self.steps += 1
            return (
                observation,
                reward,
                done,
                # self.steps > self.max_steps,
                truncation,
                info
            )

        def set_target_state(self):
            truncation = False
            if self.data.time < self.max_t:
                pos, vel, acc, _, _ = self.ideal_state_at_time()
                self.target_state = np.ravel([pos, vel, acc])
            else:
                truncation = True
            return truncation

        def ideal_state_at_time(self):
            idx = int(round(round(self.data.time, 8) * 100,8))
            return self.pos_d[idx], self.vel_d[idx], self.acc_d[idx] , self.jerk_d[idx], self.snap_d[idx]

        def _get_info(self, action, position_after, position_before):
            distance_to_target = np.linalg.norm(self.target_state[0:3] - position_after)
            # print(distance_to_target)
            observation = self._get_obs()



            reward, reward_info = self._get_rew(position_before, position_after, action)

            done = self._is_done(distance_to_target)
            info = {
                "position": self.data.qpos,
                "desired_position": self.target_state[0:3],
                "velocity": self.data.qvel,
                "desired_velocity": self.target_state[3:6],
                "done": done,
                ** reward_info,
            }
            return observation, reward, done, info

        def add_state_to_trajectory(self, action, position_after):
            observation = self._get_obs()
            current_velocity = self.data.qvel.copy()
            current_acceleration = self.data.qacc.copy()
            # print('acc: '+ str(current_acceleration))
            self.observations = np.concatenate((self.observations, observation))
            self.trajectory = np.concatenate(
                (self.trajectory, position_after, current_velocity, current_acceleration, self.target_state))
            self.actions = np.concatenate((self.actions, action))

        def goal_trajectory(self, t, radius=1.0, omega=2.0, z_amplitude=1.0, z_freq=0.5):

            x_d = radius * np.cos(omega * t) - radius
            y_d = radius * np.sin(omega * t)
            z_d = z_amplitude * np.sin(z_freq * t)
            vx_d = -radius * omega * np.sin(omega * t)
            vy_d = radius * omega * np.cos(omega * t)
            vz_d = z_amplitude * z_freq * np.cos(z_freq * t)
            return np.array([x_d, y_d, 0, vx_d, vy_d, 0])


        def reset_model(self):
            self.set_state(self.pos_d[0], self.vel_d[0])
            self.model.site_pos[self.model.site("target_state").id] = np.array([0, 0, 0])
            self.set_target_state()
            self._save_trajectory()

            self.steps = 0

            return self._get_obs()

        def _save_trajectory(self):

            if self.trajectory.size != 0:
                # print("saving trajectory")
                self.trajectory = np.reshape(self.trajectory, (-1, 18))

                file_name = trajectories_dir+"/"+self.dagger+"/trajectory_"+self.dagger+"_"+self.traj_name+".csv"
                np.savetxt(file_name, self.trajectory, delimiter=",")

                if self.first_run:

                    self.observations = np.reshape(self.observations, (-1, self.observation_space.shape[0]))
                    self.actions = np.reshape(self.trajectory, (-1, 3))
                    data = {
                        'obs': self.observations,  # (num_samples, obs_dim)
                        'act': self.actions  # (num_samples, act_dim)
                    }
                    pickle_filename = trajectories_dir+"/dagger/trajectory_expert_"+self.traj_name+".pkl"
                    with open(pickle_filename, 'wb') as f:
                        pickle.dump(data, f)
                    expert_data_filename = trajectories_dir+ "/"+self.dagger +"/trajectory_expert_"+self.traj_name+".csv"
                    np.savetxt(expert_data_filename, self.trajectory,
                               delimiter=",")
                    self.observations = np.array([])
                    self.actions = np.array([])
                    self.first_run = False

            self.trajectory = np.array([])

        def _get_obs(self):

            pos_error = self.target_state[0:3]-self.data.qpos
            self.current_pos_error = pos_error
            vel_error = self.target_state[3:6]-self.data.qvel
            acc_d = self.target_state[6:9]
            obs = np.concatenate(
                [pos_error,
                 vel_error,
                 acc_d]).ravel()

            return obs

        def _is_done(self, distance_to_target):
            if self.dagger == "dagger":
                return distance_to_target > 0.5
            return distance_to_target > 0.5

        def control_cost(self, action):
            control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
            return control_cost

        def _get_rew(self, position_before, position_after, action):

            de_weight = 1.0
            ve_weight = 0.1
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

        def check_success(self):

            max_steps_reached = self.max_steps-1 >= self.steps
            close_to_target = self.current_pos_error < 0.3
            return max_steps_reached and close_to_target
