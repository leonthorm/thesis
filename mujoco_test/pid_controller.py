import numpy as np

class PIDController:
    def __init__(self, dt, kp=1000.0, ki=10.0, kd=70.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        self.integral_error = 0
        self.previous_error = 0

    def get_action(self, state, target_state):

        error = target_state - state

        self.integral_error += error * self.dt
        derivative_error = (error - self.previous_error) / self.dt
        self.previous_error = error

        ctrl = self.kp * error + self.ki * self.integral_error + self.kd * derivative_error

        return ctrl
