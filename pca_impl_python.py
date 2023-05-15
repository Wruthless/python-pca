from multiprocessing import reduction
from turtle import color

import numpy
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

# Random seed to get consistent results.
rng = np.random.default_rng(seed=234234782384239784)

### Generating the dataset #######################################
# 3x20 Dataset
mu_vec1 = np.array([0, 0, 0])
cov_mat1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
class1_sample = rng.multivariate_normal(mu_vec1, cov_mat1, 20).T

assert class1_sample.shape == (3, 20), "[!] Dimensions are not 3x20"

# 3x20 Dataset
mu_vec2 = np.array([1, 1, 1])
cov_mat2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
class2_sample = rng.multivariate_normal(mu_vec2, cov_mat2, 20).T
assert class2_sample.shape == (3, 20), "[!] Dimensions are not 3x20"

# Remove class labels for PCA by merging the two arrays.
all_samples = np.concatenate([class1_sample, class2_sample], axis=1)
# print(all_samples)
assert all_samples.shape == (3, 40), "[-] Dimensions are not 3x40"
### End


### With the dataset we can generate the means ###########################
mean_x = np.mean(all_samples[0, :])  # row 1
mean_y = np.mean(all_samples[1, :])  # row 2
mean_z = np.mean(all_samples[2, :])  # row 3

mean_vector = np.array([[mean_x], [mean_y], [mean_z]])
# print(mean_vector)
### End


### Calculating the covariance matrix ####################################
# The covariance matrix identifies the relationship between the variables.
cov_mat = np.cov([all_samples[0, :], all_samples[1, :], all_samples[2, :]])
# print("Covariance matrix:\n", cov_mat)
### End


### Calculate the eigenvectors and relative eigenvalues ##################
eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
### End


### Sort the eigenvectors by decreasing eigenvalues #####################
# Eigenvectors are unit vectors. Therefore, to target the appropriate
# eigenvector for exclusion look at the relative eigenvector. The vector
# possessing the lowest value is the vector to exclude.

# Tuples of eigenvalues/eigenvectors.
eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:, i]) for i in range(len(eig_val_cov))]

# Sort them.
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Confirm sorting order.
# for i in eig_pairs:
#     print(i[0])
### End


### Combine eigenvectors with highest values for new matrix ###############
reduced_matrix = np.hstack((eig_pairs[0][1].reshape(3, 1), eig_pairs[1][1].reshape(3, 1)))
# print(reduced_matrix)
### End


### Projecting to new subspace ############################################
tf = reduced_matrix.T.dot(all_samples)
assert tf.shape == (2, 40)
### End

plt.plot(tf[0, 0:20], tf[1, 0:20], 'x', markersize=7, color='blue', alpha=0.5, label='class1')
plt.plot(tf[0, 20:40], tf[1, 20:40], '^', markersize=7, color='red', alpha=0.5, label='class2')
plt.xlim([-4, 4])
plt.ylim([-4, 4])
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.title('New sample set w/labels')

plt.show()
