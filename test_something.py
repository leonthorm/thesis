import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Azimuth and elevation values
arr = [2.36395143054279,
    0.7017450532681374]
azimuth = arr[0]  # Azimuth in radians (equivalent to 360 degrees)
elevation = arr[1]  # Elevation in radians (equivalent to 90 degrees)

# Convert spherical coordinates (azimuth, elevation) to Cartesian coordinates (x, y, z)
r = 1  # Radius (can be any value for visualization)
x = r * np.cos(elevation) * np.cos(azimuth)
y = r * np.sin(elevation) * np.cos(azimuth)
z = r * np.cos(elevation)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the point
ax.scatter(x, y, color='r', s=100, label=f'Azimuth: {azimuth:.2f}, Elevation: {elevation:.2f}')

# Draw a line from the origin to the point
ax.plot([0, x], [0, y],  color='b', linestyle='--')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set limits
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([0, 1])

# Add a legend
ax.legend()

# Show the plot
plt.title('3D Polar Plot of Azimuth and Elevation')
plt.show()