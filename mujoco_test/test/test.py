import numpy as np

x = np.array([])

y = np.array([1,2,3])
z = np.array([4,5,6])
# c=np.concatenate((x,y))
# print(c)
# d = np.concatenate((c,z))
# print(np.reshape(d, (-1, 3)))

s= sum((y - z) ** 2)
print(s)