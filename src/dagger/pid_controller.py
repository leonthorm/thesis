import logging

import numpy as np


class PIDController:
    def __init__(self, dt, mass, kp=20.0, ki=0.0, kd=4.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.mass = mass

        self.integral_error = 0
        self.previous_error = 0
        self.alpha = 100
    def get_action(self, obs):

        pos_error = obs[0:3]

        vel_error = obs[3:6]

        acc_des = obs[6:9]

        pd_ctrl = (
                self.kp * pos_error
                #+ self.ki * self.integral_error
                + self.kd * vel_error)
        compensate_g = self.mass * np.array([0,0,10])
        feed_forward = acc_des * self.mass + compensate_g
        return pd_ctrl + feed_forward
