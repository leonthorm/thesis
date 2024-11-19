import logging

class PIDController:
    def __init__(self, dt, kp=20.0, ki=0.0, kd=4.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        self.integral_error = 0
        self.previous_error = 0

    def get_action(self, state, target_state):
        # error = target_state - state
        pos_des = target_state[0:3]
        pos = state[0:3]
        error = pos_des - pos

        vel_des = target_state[3:6]
        vel = state[3:6]
        error_d = vel_des - vel

        # self.integral_error += error * self.dt
        # derivative_error = (error - self.previous_error) / self.dt
        # print(self.previous_error)
        # self.previous_error = error

        ctrl = (self.kp * error
                #+ self.ki * self.integral_error
                + self.kd * error_d)
        logging.info("using PID Control")
        return ctrl
