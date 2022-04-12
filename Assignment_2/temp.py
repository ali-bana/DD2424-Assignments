import numpy as np


a = np.random.uniform(0, 1, (3, 5))
b = np.array([1, 1, 1])
print(a)
print((a.T + b).T)
