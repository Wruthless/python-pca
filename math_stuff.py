import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

rng = np.random.default_rng(seed=234234782384239784)

mu_vec = np.array([0.0, 0.0, 0.0])
cov_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

class1_sample = rng.multivariate_normal(mu_vec, cov_matrix, 4).T

mean_x = np.mean(class1_sample[0, :])
mean_y = np.mean(class1_sample[1, :])
mean_z = np.mean(class1_sample[2, :])

print(class1_sample)
print()

mean_vector = np.array([mean_x, mean_y, mean_z])
print(mean_vector)

# mu_vec1 = np.array([[0.4949568], [0.46736808], [0.48120606]])
#
# X = [[0.6, 0.5, 0.7],
#      [0.4, 0.3, 0.5],
#      [0.5, 0.6, 0.4],
#      [0.5, 0.5, 0.5]]

# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(111, projection='3d')
# plt.rcParams['legend.fontsize'] = 10
#
# ax.plot()
