import logging

class PIDController:
    def __init__(self, dt, kp=20.0, ki=0.0, kd=4.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        self.integral_error = 0
        self.previous_error = 0

    def get_action(self, errors):

        pos_error = errors[0:3]

        vel_error = errors[3:6]

        ctrl = (self.kp * pos_error
                #+ self.ki * self.integral_error
                + self.kd * vel_error)
        return ctrl
