import os
import pickle
from copy import deepcopy
from datetime import datetime
import re

import gymnasium
import numpy as np
from typing import Dict, Union
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from src.util.load_traj import load_coltans_traj_and_split, get_coltrans_state_components

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.1225,
    "lookat": np.array((0.0, 0.0, 0.12250000000000005)),
}
dirname = os.path.dirname(__file__)
trajectories_dir = dirname + "/../../trajectories"


class ColtransEnv(MujocoEnv):
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
            traj_file,
            algo,
            n_robots: int,
            observation_space_size,
            dt,
            states_d,
            cable_lengths,
            xml_file: str = dirname + "/../dynamics/point_mass.xml",
            frame_skip: int = 1,
            default_camera_config: Dict[str, Union[float, int]] = {},
            healthy_reward: float = 10.0,
            reset_noise_scale: float = 0.0,
            ctrl_cost_weight: float = 0.1,
            render_mode: str = "rgb_array",
            **kwargs,
    ):

        self.x_desc = ["x_des - x[m]", "y_des - y[m]", "z_des - z[m]",
                       "vx_des - vx[m/s]", "vy_des - vy[m/s]", "vz_des - vz[m/s]",
                       "ax_des[m/s^2]", "ay_des[m/s^2]", "az_des[m/s^2]",
                       "x_other_robot[m]", "y_other_robot[m]", "z_other_robot[m]"]
        self.u_desc = ["ax[m/s^2]", "ay[m/s^2]", "az[m/s^2]"]

        observation_space = Box(low=-np.inf, high=np.inf, shape=(n_robots, observation_space_size), dtype=np.float64)
        self.algo = algo
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

        self.n_robots = n_robots
        self.cable_lengths = cable_lengths
        (self.ts,
         self.payload_pos_d, self.payload_vel_d,
         self.cable_direction_d, self.cable_ang_vel_d,
         self.robot_rot_d, self.robot_pos_d, self.robot_body_ang_vel_d, self.robot_vel_d,
         self.actions_d) = states_d
        self.target_state = [[] for i in range(n_robots)]

        self._ctrl_cost_weight = ctrl_cost_weight
        self.steps = 0
        self.max_steps = 0
        self.trajectory = np.array([])
        self.actions = np.array([])

        self.max_t = 0
        self.first_run = True
        self.traj_name = ''
        self.set_traj_name(traj_file)
        # self.current_pos_error = 0

    def set_traj(self, traj_file):
        print('target trajectory', traj_file)
        # self.ts, self.payload_pos_d, self.payload_vel_d, self.cable_direction_d, self.cable_ang_vel_d, self.robot_pos_d, self.robot_vel_d, self.robot_rot_d, self.robot_body_ang_vel_d, self.actions_d = get_coltrans_state_components(
        #     traj_file, self.n_robots, self.dt, self.cable_lengths)
        self.reset_model()
        self.set_traj_name(traj_file)
        self.max_steps = len(self.ts)

    def set_traj_name(self, traj_file):
        match = re.search(r"/([^/]+?)0[^/]*$", traj_file)
        match2 = re.search(r"([^/]+)(?=_opt\.yaml)", traj_file)
        if match:
            self.traj_name = match.group(1)
        elif match2:
            self.traj_name = match2.group(1)
        else:
            self.traj_name = traj_file

    def step(self, action):
        # pl_body = self.model.body('payload')
        # pl_data_body = self.data.body('payload')
        # pl_data_joint = self.data.joint('payload_joint')
        # pl_data_geom = self.data.geom('payload_geom')
        # quad2 = self.model.body('q2_cf2')
        # quad2_d = self.data.body('q2_cf2')
        # quad2_data = self.data.joint('q2_joint')
        # quad2_site = self.data.site('q2_imu')
        # print(self.data.time / self.dt)
        position_before = deepcopy(self.get_state())
        # self.add_state_to_trajectory(action, position_before)
        print('action: ', action)
        # print(self.target_state)
        self.do_simulation(action, self.frame_skip)
        position_after = deepcopy(self.get_state())
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

    def get_state(self):

        payload_joint = self.data.joint('payload_joint')
        payload_body = self.data.body('payload')
        payload_accelerometer = self.data.sensor('payload_linacc')

        payload_pos = payload_body.xpos
        payload_vel = payload_body.cvel[3:6]
        payload_acc = payload_accelerometer.data

        cable_direction = []
        cable_ang_vel = []
        robot_pos = []
        robot_rot = []
        robot_vel = []
        robot_ang_vel = []
        for robot in range(self.n_robots):
            robot_joint = self.data.joint(f'q{robot}_joint')
            robot_body = self.data.body(f'q{robot}_cf2')
            # cable_joint = self.data.joint(f'q{robot}_rope_J_last')

            pos = robot_body.xpos

            # cable_ang_vel.append(cable_joint.qvel)

            robot_pos.append(pos)
            # shift quaternion scalar to 4th pos
            robot_rot.append(np.roll(robot_joint.qpos[3:7], -1))
            vel = robot_joint.qvel[0:3]
            robot_vel.append(vel)
            robot_ang_vel.append(robot_joint.qvel[3:6])

            direction = pos - payload_pos
            direction = direction / np.linalg.norm(direction)

            cable_direction.append(direction)

            w_cable = np.cross(direction / self.cable_lengths[robot], (payload_vel - vel) )
            cable_ang_vel.append(w_cable)
            # robot_vel = robot_body.xvel()

        return (payload_pos, payload_vel, payload_acc,
                np.array(cable_direction), np.array(cable_ang_vel),
                np.array(robot_rot), np.array(robot_pos), np.array(robot_ang_vel), np.array(robot_vel))

    def set_target_state(self):
        truncation = False
        # TODO: truncation if trajs have different length, other robot pos
        if self.data.time < self.max_t:
            (payload_pos, payload_vel,
             cable_direction_d, cable_ang_vel,
             robot_rot, robot_pos, robot_body_ang_vel, robot_vel,
             actions) = self._ideal_state_at_time()
            for i in range(self.n_robots):
                self.target_state[i] = np.concatenate((payload_pos, payload_vel,
                                                       cable_direction_d[i], cable_ang_vel[i],
                                                       robot_rot[i], robot_pos[i], robot_body_ang_vel[i], robot_vel[i],
                                                       actions[i * 4:(i + 1) * 4]),
                                                      axis=None)
        else:
            truncation = True
        return truncation

    def _ideal_state_at_time(self):
        idx = int(round(round(self.data.time, 8) * 100, 8))
        idx = 0
        # return self.pos_d[:, idx], self.vel_d[:, idx], self.acc_d[:, idx]
        return (self.payload_pos_d[idx], self.payload_vel_d[idx],
                self.cable_direction_d[:, idx], self.cable_ang_vel_d[:, idx],
                self.robot_rot_d[:, idx], self.robot_pos_d[:, idx], self.robot_body_ang_vel_d[:, idx],
                self.robot_vel_d[:, idx],
                np.array([0.11183, 0.11183, 0.11183, 0.11183]),
                # self.actions_d[idx]
                )

    def _get_info(self, action, position_after, position_before):

        # print(distance_to_target)
        observation = self._get_obs()

        # reward, reward_info = self._get_rew(position_before, position_after, action)

        done = self._is_done(position_after)
        info = {
            "position": self.data.qpos,
            # "desired_position": self.target_state[:,0:3],
            "velocity": self.data.qvel,
            # "desired_velocity": self.target_state[:,3:6],
            "done": done,
            # **reward_info,
        }
        return observation, 0, done, info

    def add_state_to_trajectory(self, action, position):

        new_trajectory = []
        for i in range(self.n_robots):
            current_velocity = self.data.qvel[3 * i:3 * i + 3].copy()
            current_acceleration = self.data.qacc[3 * i:3 * i + 3].copy()
            new_trajectory.append(np.concatenate(
                (position[3 * i:3 * i + 3], current_velocity, current_acceleration, self.target_state[i])))
        new_trajectory = np.array(new_trajectory).ravel()
        self.trajectory = np.concatenate((self.trajectory, new_trajectory))

        observation = self._get_obs()
        new_act_obs = np.concatenate((np.round(action[0:3], 4),
                                      np.round(observation[0], 4),
                                      np.round(action[3:6], 4),
                                      np.round(observation[1], 4)
                                      ))
        self.actions = np.concatenate((self.actions, new_act_obs))

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
        self.data.eq_active[:] = 1
        warmstart = True
        if warmstart:
            for _ in range(10000):
                self.do_simulation([0,0,0,0], self.frame_skip)
            self.data.eq_active[:] = 0
            warmstart = False
            print("deactivated")
        self.data.time = 0
        self.max_t = self.ts[-1] + self.dt
        self.set_target_state()
        # self._save_trajectory()

        self.steps = 0
        self.data.eq_active[:] = 0
        return self._get_obs()

    def _save_trajectory(self):
        if len(self.trajectory) != 0:

            state_space = 6
            action_space = 3
            self.trajectory = np.reshape(self.trajectory, (-1, (state_space + action_space) * 2 * self.n_robots))
            header = ""
            for i in range(self.n_robots):
                header += f"x_robot{i + 1},y_robot{i + 1},z_robot{i + 1},x_vel_robot{i + 1},y_vel_robot{i + 1},z_vel_robot{i + 1},x_acc_robot{i + 1},y_acc_robot{i + 1},z_acc_robot{i + 1},x_des_robot{i + 1},y_des_robot{i + 1},z_des_robot{i + 1},x_vel_des_robot{i + 1},y_vel_des_robot{i + 1},z_vel_des_robot{i + 1},x_acc_des_robot{i + 1},y_acc_des_robot{i + 1},z_acc_des_robot{i + 1},"
            file_name = f"{trajectories_dir}/{self.algo}/dbcbs/trajectory_{self.algo}_{self.traj_name}.csv"
            np.savetxt(file_name, self.trajectory, delimiter=",", header=header)

            if self.first_run:
                expert_data_filename = f"{trajectories_dir}/{self.algo}/dbcbs/trajectory_expert_{self.algo}_{self.traj_name}.csv"
                np.savetxt(expert_data_filename, self.trajectory, delimiter=",", header=header)

                # self.actions = np.reshape(self.actions, (-1, 30))
                # actions_filename = f"{trajectories_dir}/{self.algo}/dbcbs/actions.csv"
                # np.savetxt(actions_filename, self.actions, delimiter=",")

                self.first_run = False

            self.trajectory = np.array([])

    def _get_obs(self):

        (payload_pos, payload_vel, payload_acc,
        cable_direction, cable_ang_vel,
        robot_rot, robot_pos, robot_ang_vel, robot_vel) = self.get_state()

        payload_acc -= np.array([0,0,9.81])
        observation_arrays = [
            np.full((self.n_robots, 3), payload_pos),
            np.full((self.n_robots, 3), payload_vel),
            np.full((self.n_robots, 3), payload_acc),
            cable_direction,
            cable_ang_vel,
            robot_rot,
            robot_pos,
            robot_ang_vel,
            robot_vel
        ]

        observations = np.concatenate(observation_arrays, axis=1)

        other_robot_pos = [np.concatenate((robot_pos[:i], robot_pos[i+1:])).ravel() for i in range(self.n_robots)]

        robot_id = [[i] for i in range(self.n_robots)]

        observations = np.concatenate((observations, np.array(other_robot_pos), self.target_state, robot_id), axis=1)

        print('pp: ', payload_pos)
        print('pp d: ', self.target_state[0][0:3])
        print('vp: ', payload_vel)
        print('vp d: ', self.target_state[0][3:6])
        print('qc: ', cable_direction)
        print('qc d: ', self.target_state[0][6:9])
        print('wc: ', cable_ang_vel)
        print('wc d: ', self.target_state[0][9:12])
        print('quat: ', robot_rot)
        print('quat d: ', self.target_state[0][12:16])
        print('p: ', robot_pos)
        print('p d: ', self.target_state[0][16:19])
        print('wr: ', robot_ang_vel)
        print('wr d: ', self.target_state[0][19:22])
        print('vr: ', robot_vel)
        print('vr d: ', self.target_state[0][22:25])
        print('action d:', self.target_state[0][25:29])
        return observations

    def _is_done(self, position_after):

        done = False
        (payload_pos, _, _,
        _, _,
        _, _, _, _) = position_after
        # if payload distance to referenece is to high
        payload_pos_error = np.linalg.norm(payload_pos - self.target_state[0][0:3])
        if payload_pos_error > 1:
            done = True
        return False

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

        reward = distance_reward * de_weight - ctrl_cost + velocity_error * ve_weight

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

        max_steps_reached = self.max_steps - 1 >= self.steps
        close_to_target = self.current_pos_error < 0.3
        return max_steps_reached and close_to_target
