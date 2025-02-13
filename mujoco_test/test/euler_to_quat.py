import numpy as np
from scipy.spatial.transform import Rotation as R

euler_a_1 = np.array([0., -2.356194, 0.])
euler_a_2 = np.array([-0.785398, 0., -0.785398])
print(R.from_euler('xyz', euler_a_1, degrees=False).as_quat(scalar_first=True))
print(R.from_euler('xyz', euler_a_2, degrees=False).as_quat(scalar_first=True))