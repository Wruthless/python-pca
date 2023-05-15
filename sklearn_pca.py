from sklearn.decomposition import PCA as sklearnPCA
import numpy as np
from matplotlib import pyplot as plt

# Seed for consistent results
rng = np.random.default_rng(seed=234234782384239784)

# Same data point generation as in the python implementation.
mu_vec1 = np.array([0, 0, 0])
cov_mat1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
class1_sample = rng.multivariate_normal(mu_vec1, cov_mat1, 20).T

mu_vec2 = np.array([1, 1, 1])
cov_mat2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
class2_sample = rng.multivariate_normal(mu_vec2, cov_mat2, 20).T

all_samples = np.concatenate([class1_sample, class2_sample], axis=1)

mean_x = np.mean(all_samples[0, :])  # row 1
mean_y = np.mean(all_samples[1, :])  # row 2
mean_z = np.mean(all_samples[2, :])  # row 3

mean_vector = np.array([[mean_x], [mean_y], [mean_z]])

sklearn_pca = sklearnPCA(n_components=2)
sklearn_transf = sklearn_pca.fit_transform(all_samples.T)
sklearn_transf = sklearn_transf * (-1)

plt.plot(sklearn_transf[0:20, 0], sklearn_transf[0:20, 1], 'x', markersize=7, color='blue', alpha=0.5, label='class1')
plt.plot(sklearn_transf[20:40, 0], sklearn_transf[20:40, 1], '^', markersize=7, color='red', alpha=0.5, label='class2')
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.xlim([-4, 4])
plt.ylim([-4, 4])
plt.legend()
plt.title('Transformed samples via sklearn.decomposition.PCA')
plt.show()
