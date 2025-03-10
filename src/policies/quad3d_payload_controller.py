import sys
import os

from src.util.helper import deconstruct_obs

sys.path.append("./")
# sys.path.append('../')
import numpy as np
import math
import rowan as rn

# import cvxpy as cp
import time
import cffirmware
import rowan
import yaml
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import cvxpy as cp


class Quad3dPayloadController:
    def __init__(self, robotparams, gains):
        self.gains = gains
        self.mi = robotparams["mi"]
        self.mp = robotparams["mp"]
        self.payloadType = robotparams["payloadType"]
        self.robot_radius = robotparams["robot_radius"]
        self.mu_planned = []
        self.mu_desired = []
        # start_idx: to change the updateState according to the type of payload
        self.t2t = 0.006
        arm_length = 0.046
        self.arm = 0.707106781 * arm_length
        u_nominal = self.mi * 9.81 / 4
        self.num_robots = robotparams["num_robots"]
        self.B0 = u_nominal * np.array(
            [
                [1, 1, 1, 1],
                [-self.arm, -self.arm, self.arm, self.arm],
                [-self.arm, self.arm, self.arm, -self.arm],
                [-self.t2t, self.t2t, -self.t2t, self.t2t],
            ]
        )
        self.B0_inv = np.linalg.inv(self.B0)
        self.Mq = (self.mp) * np.eye(3)
        # self.invMq = np.linalg.inv(self.Mq)
        kpos_p, kpos_d, kpos_i = gains[0]

        kc_p, kc_d, kc_i = gains[1]

        kth_p, kth_d, kth_i = gains[2]
        kp_limit, kd_limit, ki_limit = gains[3]
        lambdaa = gains[4]
        self.Ji = np.diag(robotparams["Ji"])
        self.nocableTracking = robotparams["nocableTracking"]
        self.leePayload = cffirmware.controllerLeePayload_t()
        cffirmware.controllerLeePayloadInit(self.leePayload)
        self.team_state = dict()
        self.team_ids = [i for i in range(self.num_robots)]
        self.leePayload.mp = self.mp
        self.l = robotparams["l"]
        self.leePayload.en_qdidot = 1  # 0: disable, 1: provide references
        self.leePayload.mass = self.mi
        if self.payloadType == "point":
            self.leePayload.en_accrb = (
                0  # TODO: don't forget to change this for the rigid case
            )
            self.leePayload.gen_hp = 1  # TODO: don't forget to change this after updating the firmware for rigid case
        elif self.payloadType == "rigid":
            self.Jp = robotparams["Jp"]
            self.leePayload.en_accrb = (
                1  # TODO: don't forget to change this for the rigid case
            )
            self.leePayload.gen_hp = 1  # TODO: don't forget to change this after updating the firmware for rigid case, I don't think we use this anymore
        if self.nocableTracking:
            self.leePayload.formation_control = 0  # 0: disable, 1:set mu_des_prev (regularization), 3: planned formations (qi refs)
        else:
            self.leePayload.formation_control = 3  # 0: disable, 1:set mu_des_prev (regularization), 3: planned formations (qi refs)
        # exit()
        self.leePayload.lambda_svm = 1000
        self.leePayload.radius = self.robot_radius
        self.leePayload.lambdaa = lambdaa
        self.leePayload.Kpos_P.x = kpos_p
        self.leePayload.Kpos_P.y = kpos_p
        self.leePayload.Kpos_P.z = kpos_p

        self.leePayload.Kpos_D.x = kpos_d
        self.leePayload.Kpos_D.y = kpos_d
        self.leePayload.Kpos_D.z = kpos_d

        self.leePayload.Kpos_I.x = kpos_i
        self.leePayload.Kpos_I.y = kpos_i
        self.leePayload.Kpos_I.z = kpos_i

        self.leePayload.Kpos_P_limit = kp_limit
        self.leePayload.Kpos_I_limit = kd_limit
        self.leePayload.Kpos_D_limit = ki_limit
        if self.payloadType == "rigid":
            self.attP = robotparams["attP"]
            kp_pth, kd_pth = gains[5]
            self.leePayload.Kprot_P.x = kp_pth
            self.leePayload.Kprot_P.y = kp_pth
            self.leePayload.Kprot_P.z = kp_pth
            self.leePayload.Kprot_D.x = kd_pth
            self.leePayload.Kprot_D.y = kd_pth
            self.leePayload.Kprot_D.z = kd_pth
        self.leePayload.KR.x = kth_p
        self.leePayload.KR.y = kth_p
        self.leePayload.KR.z = kth_p
        self.leePayload.Komega.x = kth_d
        self.leePayload.Komega.y = kth_d
        self.leePayload.Komega.z = kth_d
        self.leePayload.KI.x = kth_i
        self.leePayload.KI.y = kth_i
        self.leePayload.KI.z = kth_i

        self.leePayload.K_q.x = kc_p
        self.leePayload.K_q.y = kc_p
        self.leePayload.K_q.z = kc_p
        self.leePayload.K_w.x = kc_d
        self.leePayload.K_w.y = kc_d
        self.leePayload.K_w.z = kc_d
        self.leePayload.KqIx = kc_i
        self.leePayload.KqIy = kc_i
        self.leePayload.KqIz = kc_i
        self.control = cffirmware.control_t()
        # allocate desired state
        setpoint_ = cffirmware.setpoint_t()
        self.setpoint = self.__setTrajmode(setpoint_)
        self.sensors = cffirmware.sensorData_t()
        self.state = cffirmware.state_t()
        num_robots = robotparams["num_robots"]
        self.state.num_uavs = num_robots

    def __setTrajmode(self, setpoint):
        """This function sets the trajectory modes of the controller"""
        setpoint.mode.x = cffirmware.modeAbs
        setpoint.mode.y = cffirmware.modeAbs
        setpoint.mode.z = cffirmware.modeAbs
        setpoint.mode.quat = cffirmware.modeAbs
        setpoint.mode.roll = cffirmware.modeDisable
        setpoint.mode.pitch = cffirmware.modeDisable
        setpoint.mode.yaw = cffirmware.modeDisable
        return setpoint

    def __updateSensor(self, state, i):
        """This function updates the sensors signals"""
        _, _, _, w = self.__getUAVSt(state, i)
        self.sensors.gyro.x = np.degrees(w[0])  # deg/s
        self.sensors.gyro.y = np.degrees(w[1])  # deg/s
        self.sensors.gyro.z = np.degrees(w[2])  # deg/s

    def __computeAcc(self, state, actions, tick):
        ap_ = np.zeros(
            3,
        )
        for k, i in enumerate(self.team_ids):
            action = actions[4 * k: 4 * k + 4]
            q = state[
                9 + 6 * self.num_robots + 7 * k: 9 + 6 * self.num_robots + 7 * k + 4
                ]
            qc = state[9 + 6 * k: 9 + 6 * k + 3]
            wc = state[9 + 6 * k + 3: 9 + 6 * k + 6]
            q_rn = [q[3], q[0], q[1], q[2]]
            control = self.B0 @ action
            th = control[0]
            fu = np.array([0, 0, th])
            u_i = rn.rotate(q_rn, fu)
            qcqcT = qc.reshape(3, 1) @ qc.reshape(1, 3)
            ap_ += qcqcT @ u_i - self.mi * self.l[k] * np.dot(wc, wc) * qc
            self.Mq += self.mi * qcqcT
        ap = np.linalg.inv(self.Mq) @ (ap_)

        if tick > 0:
            ap = np.linalg.inv(self.Mq) @ (ap_)  # - np.array([0,0,9.81])
        return ap

    def __comuteAngAcc(self, state, actions, ap, i, wpdot=None):
        if self.payloadType == "point":
            action = actions[4 * i: 4 * i + 4]
            q = state[
                9 + 6 * self.num_robots + 7 * i: 9 + 6 * self.num_robots + 7 * i + 4
                ]
            qc = state[9 + 6 * i: 9 + 6 * i + 3]
            wc = state[9 + 6 * i + 3: 9 + 6 * i + 6]
            q_rn = [q[3], q[0], q[1], q[2]]
            control = self.B0 @ action
            th = control[0]
            fu = np.array([0, 0, th])
            apgrav = ap + np.array([0, 0, 9.81])
            u_i = rn.rotate(q_rn, fu)
            qcqcT = qc.reshape(3, 1) @ qc.reshape(3, 1).T
            wcdot = 1 / self.l[i] * np.cross(qc, apgrav) - (
                    1 / (self.mi * self.l[i])
            ) * np.cross(qc, u_i)

        return wcdot

    def __computeUAVwd(self, states, actions, i):
        q = states[
            16 + 6 * self.num_robots + 7 * i: 16 + 6 * self.num_robots + 7 * i + 4
            ]
        q_rn = [q[3], q[0], q[1], q[2]]
        w = states[
            16 + 6 * self.num_robots + 7 * i + 4: 16 + 6 * self.num_robots + 7 * i + 7
            ]

        control = self.B0 @ actions[4 * i: 4 * i + 4]
        w_dot = np.linalg.inv(self.Ji) @ (tau - skew(w) @ self.Ji @ w)
        return w_dot

    def __updateDesState(self, actions_d, states_d, state, compAcc, a_ref, tick):
        self.setpoint.position.x = states_d[0]  # m
        self.setpoint.position.y = states_d[1]  # m
        self.setpoint.position.z = states_d[2]  # m
        ap = a_ref
        if self.payloadType == "point":
            start_idx = 0
            rig_idx = 0
        if compAcc:
            ap = self.__computeAcc(states_d, actions_d, tick)

        states_d[start_idx + 6: start_idx + 9] = ap
        self.setpoint.velocity.x = states_d[start_idx + 3]  # m/s
        self.setpoint.velocity.y = states_d[start_idx + 4]  # m/s
        self.setpoint.velocity.z = states_d[start_idx + 5]  # m/s
        self.setpoint.acceleration.x = states_d[
            start_idx + 6
            ]  # m/s^2 update this to be computed from model
        self.setpoint.acceleration.y = states_d[
            start_idx + 7
            ]  # m/s^2 update this to be computed from model
        self.setpoint.acceleration.z = states_d[
            start_idx + 8
            ]  # m/s^2 update this to be computed from model

        mu_planned_tmp = []
        mu_des_tmp = []
        self.mu_planned = []
        tensions = 0
        grav = np.array([0, 0, 9.81])
        self.kpos_p, self.kpos_d, kpos_i = self.gains[0]

        self.posp_e = states_d[0:3] - state[0:3]
        self.velp_e = states_d[start_idx + 3: start_idx + 6] - state[3:6]
        accp = ap + grav
        self.F_ref = self.mp * (
                accp + self.kpos_p * (self.posp_e) + self.kpos_d * (self.velp_e)
        )
        mu_planned_sum = np.zeros(
            3,
        )
        qi_mat = np.zeros((3, self.num_robots))

        second_term = np.zeros(
            3,
        )
        for k, i in enumerate(self.team_ids):
            qc = states_d[start_idx + 9 + 6 * i: start_idx + 9 + 6 * i + 3]
            wc = states_d[start_idx + 9 + 6 * i + 3: start_idx + 9 + 6 * i + 6]
            qi_mat[0:3, k] = -qc
            # second_term +=  self.mi*self.l[i]*wc.dot(wc)*qc

        second_term += self.F_ref

        # Construct the problem.
        n = self.num_robots
        T_vec = cp.Variable(n)
        objective = cp.Minimize(
            0.001 * cp.sum_squares(T_vec)
            + cp.sum_squares((qi_mat @ T_vec - self.F_ref))
        )
        constraints = [
            0.001 * np.ones(
                n,
            )
            <= T_vec
        ]
        prob = cp.Problem(objective, constraints)
        # The optimal objective value is returned by `prob.solve()`.
        result = prob.solve()
        T_vec = T_vec.value
        # T_vec = np.linalg.pinv(qi_mat)@second_term
        for k, i in enumerate(self.team_ids):
            qc = states_d[start_idx + 9 + 6 * i: start_idx + 9 + 6 * i + 3]
            wc = states_d[start_idx + 9 + 6 * i + 3: start_idx + 9 + 6 * i + 6]
            qc_dot = np.cross(wc, qc)
            action = actions_d[4 * k: 4 * k + 4]
            control = self.B0 @ action
            self.leePayload.tau_ff.x = control[1]
            self.leePayload.tau_ff.y = control[2]
            self.leePayload.tau_ff.z = 0.
            w_des = states_d[start_idx + 9 + 6 * self.num_robots + 4: start_idx + 9 + 6 * self.num_robots + 7]
            self.leePayload.omega_r.x = w_des[0]
            self.leePayload.omega_r.y = w_des[1]
            self.leePayload.omega_r.z = w_des[2]
            mu_planned = -T_vec[k] * qc
            mu_planned_tmp.extend(mu_planned.tolist())
            mu_planned_sum += mu_planned
            wcdot = self.__comuteAngAcc(states_d, actions_d, ap, i)
            cffirmware.set_setpoint_qi_ref(
                self.setpoint,
                k,
                k,
                mu_planned[0],
                mu_planned[1],
                mu_planned[2],
                qc_dot[0],
                qc_dot[1],
                qc_dot[2],
                wcdot[0],
                wcdot[1],
                wcdot[2],
            )
            # cffirmware.set_setpoint_qi_ref(self.setpoint, k, k,  mu_planned[0], mu_planned[1], mu_planned[2], qc_dot[0], qc_dot[1], qc_dot[2], 0, 0, 0)
        self.mu_planned.append(mu_planned_tmp)

    def __getUAVSt(self, state, i):
        if self.payloadType == "point":
            cable_start_idx = 6

        l = self.l[i]
        qc = state[cable_start_idx + 6 * i: cable_start_idx + 6 * i + 3]
        wc = state[cable_start_idx + 6 * i + 3: cable_start_idx + 6 * i + 6]
        quat = state[
               cable_start_idx
               + 6 * self.num_robots
               + 7 * i: cable_start_idx
                        + 6 * self.num_robots
                        + 7 * i
                        + 4
               ]
        w = state[
            cable_start_idx
            + 6 * self.num_robots
            + 7 * i
            + 4: cable_start_idx
                 + 6 * self.num_robots
                 + 7 * i
                 + 7
            ]

        qc_dot = np.cross(wc, qc)
        pos = np.array(state[0:3]) - l * qc
        vel = np.array(state[3:6]) - l * qc_dot
        return pos, vel, quat, w

    def __updateState(self, state, i):
        start_idx = 3
        if self.payloadType == "rigid":
            self.state.payload_quat.x = state[3]
            self.state.payload_quat.y = state[4]
            self.state.payload_quat.z = state[5]
            self.state.payload_quat.w = state[6]
            start_idx = 7
            self.state.payload_omega.x = state[start_idx + 3]
            self.state.payload_omega.y = state[start_idx + 4]
            self.state.payload_omega.z = state[start_idx + 5]
        else:
            self.state.payload_quat.x = np.nan
            self.state.payload_quat.y = np.nan
            self.state.payload_quat.z = np.nan
            self.state.payload_quat.w = np.nan

        self.state.payload_pos.x = state[0]  # m
        self.state.payload_pos.y = state[1]  # m
        self.state.payload_pos.z = state[2]  # m
        self.state.payload_vel.x = state[start_idx]  # m/s
        self.state.payload_vel.y = state[start_idx + 1]  # m/s
        self.state.payload_vel.z = state[start_idx + 2]  # m/s

        pos, vel, quat, w = self.__getUAVSt(state, i)
        self.state.position.x = pos[0]  # m
        self.state.position.y = pos[1]  # m
        self.state.position.z = pos[2]  # m
        self.state.velocity.x = vel[0]  # m/s
        self.state.velocity.y = vel[1]  # m/s
        self.state.velocity.z = vel[2]  # m/s

        rpy_state = rn.to_euler([quat[3], quat[0], quat[1], quat[2]], convention="xyz")

        self.state.attitude.roll = np.degrees(rpy_state[0])
        self.state.attitude.pitch = np.degrees(-rpy_state[1])
        self.state.attitude.yaw = np.degrees(rpy_state[2])
        self.state.attitudeQuaternion.x = quat[0]
        self.state.attitudeQuaternion.y = quat[1]
        self.state.attitudeQuaternion.z = quat[2]
        self.state.attitudeQuaternion.w = quat[3]
        self.mu_desired = self.leePayload.desVirtInp

    def __updateNeighbors(self, state):
        for k, i in enumerate(self.team_ids):
            pos, _, _, _ = self.__getUAVSt(state, i)
            cffirmware.state_set_position(self.state, k, k, pos[0], pos[1], pos[2])
            cffirmware.controller_lee_payload_set_attachement(
                self.leePayload, k, k, 0, 0, 0
            )

    def controllerLeePayload(
            self, actions_d, states_d, state, tick, my_id, compAcc, a_ref
    ):
        self.team_ids.remove(my_id)
        self.team_ids.insert(0, my_id)
        self.__updateDesState(actions_d, states_d, state, compAcc, a_ref, tick)
        self.__updateState(state, my_id)
        self.__updateSensor(state, my_id)
        self.__updateNeighbors(state)
        cffirmware.controllerLeePayload(
            self.leePayload, self.control, self.setpoint, self.sensors, self.state, tick
        )
        self.leePayload.payload_vel_prev.x = state[3]
        self.leePayload.payload_vel_prev.y = state[4]
        self.leePayload.payload_vel_prev.z = state[5]

        control = np.array(
            [
                self.leePayload.thrustSI,
                self.control.torque[0],
                self.control.torque[1],
                self.control.torque[2],
            ]
        )
        # print("errors and gains: ",self.posp_e, self.kpos_p)
        # print("errors and gains: ",self.velp_e, self.kpos_d)

        u = self.B0_inv @ control
        return u.tolist()
