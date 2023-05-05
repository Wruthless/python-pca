import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

rng = np.random.default_rng(seed=234234782384239784)

# Mean Vector
mu_vec1 = np.array([0, 0, 0])
# Covariance Matrix
cov_mat1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Multivariate normal distribution / Probability Density
class1_sample = rng.multivariate_normal(mu_vec1, cov_mat1, 20).T

assert class1_sample.shape == (3, 20), "[!] Dimensions are not 3x20"

mu_vec2 = np.array([1, 1, 1])
cov_mat2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
class2_sample = rng.multivariate_normal(mu_vec2, cov_mat2, 20).T
assert class2_sample.shape == (3, 20), "[!] Dimensions are not 3x20"

all_samples = np.concatenate([class1_sample, class2_sample], axis=1)
assert all_samples.shape == (3, 40), "[-] Dimensions are not 3x40"

mean_x = np.mean(all_samples[0, :])  # row 1
mean_y = np.mean(all_samples[1, :])  # row 2
mean_z = np.mean(all_samples[2, :])  # row 3

# The mean vector is what centers the dataset to the axes before
# the principal component calculations are performed.
mean_vector = np.array([[mean_x], [mean_y], [mean_z]])

print('Mean vector:\n', mean_vector)

# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')
# plt.rcParams['legend.fontsize'] = 10
#
# ax.plot(class1_sample[0, :], class1_sample[1, :], class1_sample[2, :],
#         'o', markersize=8, color='blue', alpha=0.5, label='class1')
#
# ax.plot(class2_sample[0, :], class2_sample[1, :], class2_sample[2, :],
#         '^', markersize=8, color='red', alpha=0.5, label='class2')
#
#
# plt.title('Samples for class 1 and class 2')
# ax.legend(loc='upper right')
#
# plt.show()
