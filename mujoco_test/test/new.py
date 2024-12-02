import numpy as np

array = np.array([10, 20, 30, 40, 50])

# Value to find
value = 30

# Find the index
indices = np.where(array == value)

print(indices[0])