import numpy as np
from matplotlib import pyplot as plt

from src.thrifty.algos.thriftydagger import thrifty

# expert = np.loadtxt("trajectories/dagger/trajectory_2024-12-11 23:19:01.307048.csv", delimiter=",")
# trajectory = np.loadtxt("../../src/thrifty/trajectories/trajectory_2024-12-09 18:56:15.412951.csv", delimiter=",")
# trajectory = np.loadtxt("trajectories/dagger/trajectory_2024-12-11 23:19:16.928623.csv", delimiter=",")


# x1, y1, z1, x_vel, y_vel, z_vel = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], trajectory[:, 3], trajectory[:, 4], trajectory[:, 5]
# x_d, y_d, z_d, x_vel_d, y_vel_d, z_vel_d = trajectory[:, 6], trajectory[:, 7], trajectory[:, 8], trajectory[:, 9], trajectory[:, 10], trajectory[:, 11]
# x_e, y_e, z_e, x_vel, y_vel, z_vel = expert[:, 0], expert[:, 1], expert[:, 2], expert[:, 3], expert[:, 4], expert[:, 5]




def plot(to_plot, thrifty, trajectorie_type_dict, trajectory_type, trajectory, expert=None):

    x_e, y_e, z_e, x_e_vel, y_e_vel, z_e_vel, x_e_acc, y_e_acc, z_e_acc = (
        expert[:, 0], expert[:, 1], expert[:, 2],
        expert[:, 3], expert[:, 4], expert[:, 5],
        expert[:, 6], expert[:, 7], expert[:, 8]
    )

    x, y, z, x_vel, y_vel, z_vel, x_acc, y_acc, z_acc = (
        trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
        trajectory[:, 3], trajectory[:, 4], trajectory[:, 5],
        trajectory[:, 6],trajectory[:, 7],trajectory[:, 8]
    )

    x_d, y_d, z_d, x_vel_d, y_vel_d, z_vel_d, x_acc_d, y_acc_d, z_acc_d = (
        trajectory[:, 9], trajectory[:, 10], trajectory[:, 11],
        trajectory[:, 12], trajectory[:, 13], trajectory[:, 14],
        trajectory[:, 15], trajectory[:, 16], trajectory[:, 17]
    )

    if to_plot == 1:
        plt.figure(figsize=(10, 10))
        plt.plot(x, y, color='blue', label='policy trajectory', alpha=0.7)
        plt.plot(x_d, y_d, color='green', label='ideal trajectory', alpha=0.7)
        plt.plot(x_e, y_e, color='red', label='expert trajectory', alpha=0.7)
        plt.plot(x_d[0],y_d[0], marker="^",color='black', label='start_pos')
        plt.title(f'{trajectorie_type_dict[trajectory_type]} point mass trajectories')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.show()

    elif to_plot == 2:
        plt.figure(figsize=(10, 10))
        plt.plot(x, z, color='blue', label='policy trajectory', alpha=0.7)
        plt.plot(x_d, z_d, color='green', label='ideal trajectory', alpha=0.7)
        plt.plot(x_e, z_e, color='red', label='expert trajectory', alpha=0.7)
        plt.plot(x_d[0],z_d[0], marker="^",color='black', label='start_pos')
        plt.title(f'{trajectorie_type_dict[trajectory_type]} point mass trajectories with z-axis')
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.legend()
        plt.grid(True)
        plt.show()

    elif to_plot == 3:
        pos_error = trajectory[:, 9:12] - trajectory[:, 0:3]
        pos_error_norm = np.linalg.norm(pos_error, axis=1)
        expert_error = expert[:, 9:12] - expert[:, 0:3]
        expert_error_norm = np.linalg.norm(expert_error, axis=1)

        pos_error_x = pos_error[:, 0]
        pos_error_y = pos_error[:, 1]
        pos_error_z = pos_error[:, 2]

        fig, ax = plt.subplots(2, 1, figsize=(10, 10))

        ax[0].plot(pos_error_norm, color='blue', label='state error norm')
        ax[0].plot(expert_error_norm, color='red', label='expert error norm')
        ax[0].set_title(f'{trajectorie_type_dict[trajectory_type]} State Error Norm')
        ax[0].legend()
        ax[0].set_ylabel('State Error Norm')
        ax[0].set_xlabel('Step')
        ax[0].grid(True)

        ax[1].plot(pos_error_x, color='red', label='state error x')
        ax[1].plot(pos_error_y, color='blue', label='state error y')
        ax[1].plot(pos_error_z, color='green', label='state error z')
        ax[1].set_title(f'{trajectorie_type_dict[trajectory_type]} State Error per Axis')
        ax[1].legend()
        ax[1].set_ylabel('State Error')
        ax[1].set_xlabel('Step')
        ax[1].grid(True)

        plt.tight_layout()
        plt.show()

    elif to_plot == 4:
        vel_error = trajectory[:, 9:12] - trajectory[:, 3:6]
        vel_error_norm = np.linalg.norm(vel_error, axis=1)
        expert_error = expert[:, 9:12] - expert[:, 3:6]
        expert_error_norm = np.linalg.norm(expert_error, axis=1)

        vel_error_x = vel_error[:, 0]
        vel_error_y = vel_error[:, 1]
        vel_error_z = vel_error[:, 2]

        fig, ax = plt.subplots(2, 1, figsize=(10, 10))

        ax[0].plot(vel_error_norm, color='blue', label='velocity error norm')
        ax[0].plot(expert_error_norm, color='red', label='expert error norm')
        ax[0].set_title(f'{trajectorie_type_dict[trajectory_type]} Velocity Error Norm')
        ax[0].legend()
        ax[0].set_ylabel('Velocity Error Norm')
        ax[0].set_xlabel('Step')
        ax[0].grid(True)

        ax[1].plot(vel_error_x, color='red', label='velocity error x')
        ax[1].plot(vel_error_y, color='blue', label='velocity error y')
        ax[1].plot(vel_error_z, color='green', label='velocity error z')
        ax[1].set_title('Velocity Error per Axis')
        ax[1].legend()
        ax[1].set_ylabel('State Error')
        ax[1].set_xlabel('Step')
        ax[1].grid(True)

        plt.tight_layout()
        plt.show()

    elif to_plot == 5:
        plt.figure(figsize=(10, 10))
        pos_difference= np.linalg.norm(trajectory[:, 0:3]-expert[:, 0:3], axis=1)

        plt.title(f'{trajectorie_type_dict[trajectory_type]}  state difference')
        plt.plot(pos_difference, label='policy expert difference')
        plt.xlabel('step')
        plt.ylabel('difference')
        plt.legend()
        plt.grid(True)
        plt.show()

    elif to_plot == 6:
        fig, ax = plt.subplots(3, 1, figsize=(10, 15))
        ax[0].plot(x_d, color='red', linestyle='dotted', label='ideal x')
        ax[0].plot(y_d, color='blue', linestyle='dotted', label='ideal y')
        ax[0].plot(z_d, color='green', linestyle='dotted', label='ideal z')
        ax[0].plot(x_e, color='red', label='expert x')
        ax[0].plot(y_e, color='blue', label='expert y')
        ax[0].plot(z_e, color='green', label='expert z')
        ax[0].set_title(f'{trajectorie_type_dict[trajectory_type]}  Position per Axis')
        ax[0].legend()
        ax[0].set_ylabel('Position')
        ax[0].set_xlabel('Step')
        ax[0].grid(True)

        ax[1].plot(x_vel_d, color='red', linestyle='dotted', label='ideal x_vel')
        ax[1].plot(y_vel_d, color='blue', linestyle='dotted', label='ideal y_vel')
        ax[1].plot(z_vel_d, color='green', linestyle='dotted', label='ideal z_vel')
        ax[1].plot(x_e_vel, color='red', label='expert x_vel')
        ax[1].plot(y_e_vel, color='blue', label='expert y_vel')
        ax[1].plot(z_vel_d, color='green', label='expert z_vel')
        ax[1].set_title('Velocity per Axis')
        ax[1].legend()
        ax[1].set_ylabel('Velocity')
        ax[1].set_xlabel('Step')
        ax[1].grid(True)

        ax[2].plot(x_acc_d, color='red', linestyle='dotted', label='ideal x acc')
        ax[2].plot(y_acc_d, color='blue', linestyle='dotted', label='ideal y acc')
        ax[2].plot(z_acc_d, color='green', linestyle='dotted', label='ideal z acc')
        ax[2].plot(x_e_acc, color='red', label='expert x acc')
        ax[2].plot(y_e_acc, color='blue', label='expert y acc')
        ax[2].plot(z_e_acc, color='green', label='expert z acc')
        ax[2].set_title('Acceleration per Axis')
        ax[2].legend()
        ax[2].set_ylabel('Acceleration')
        ax[2].set_xlabel('Step')
        ax[2].grid(True)

        plt.tight_layout()
        plt.show()


def get_metrics(thrifty=False, trajectory=None):
    if trajectory is None:
        trajectory = np.loadtxt("trajectories/dagger/trajectory_dagger_wave.csv", delimiter=",")

    if thrifty:
        print('thrifty')
    else:
        print('dagger')
    state_error = np.linalg.norm(trajectory[:, 0:3]-trajectory[:, 9:12], axis=1)
    vel_error = np.linalg.norm(trajectory[:, 3:6]-trajectory[:, 12:15], axis=1)
    state_error_mean = np.std(state_error)
    state_error_std = np.mean(state_error)
    vel_error_mean = np.std(vel_error)
    vel_error_std = np.mean(vel_error)

    print("state_error mean: ", np.mean(state_error))
    print("state_error std: ", np.std(state_error))
    print("vel_error mean: ", np.mean(vel_error))
    print("vel_error std: ", np.std(vel_error))

    return state_error_mean, state_error_std, vel_error_mean, vel_error_std

def plot_all(thrifty, trajectory, expert):
    for i in range(1, 5):
        plot(i,thrifty,trajectory, expert)
    if not thrifty:
        plot(5,thrifty,trajectory, expert)


def load_trajectory(trajectories, trajectory_type, thrifty):
    dagger_csv = f"trajectory_dagger_{trajectories[trajectory_type]}.csv"
    thrifty_csv = f"trajectory_thrifty_{trajectories[trajectory_type]}.csv"
    # thrifty_csv = "thrifty/trajectory_thrifty.csv"
    expert_csv = f"trajectory_expert_{trajectories[trajectory_type]}.csv"
    if thrifty:
        trajectory = np.loadtxt("trajectories/thrifty/" + thrifty_csv, delimiter=",")
    else:
        trajectory = np.loadtxt("trajectories/dagger/" + dagger_csv, delimiter=",")
    expert = np.loadtxt("trajectories/dagger/" + expert_csv, delimiter=",")


    return trajectory, expert

if __name__ == '__main__':




    to_plot = {
        1: "point mass trajectories",
        2: "point mass trajectories with z-axis",
        3: "state error",
        4: "velocity error",
        5: "state difference",
        6: "plot all components"
    }
    trajectorie_type_dict = {
        1: "circle",
        2: "figure8",
        3: "helix",
        4: "lissajous",
        5: "radial_oscillation",
        6: "wave",
    }
    thrifty = True
    trajectory_type = 1

    trajectory, expert = load_trajectory(trajectorie_type_dict, trajectory_type, thrifty)
    plot(3, thrifty, trajectorie_type_dict, trajectory_type, trajectory, expert)
    get_metrics(thrifty, trajectory)
    # plot_all(thrifty, trajectory, expert)

