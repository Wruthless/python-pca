from sklearn.decomposition import PCA
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

# mpl and sklearn PCA.
mlab_pca = PCA(n_components=2)
mlab_pca.fit(all_samples.T)
transformed_data = mlab_pca.transform(all_samples.T)

transformed_class1 = transformed_data[:20, :]
transformed_class2 = transformed_data[20:40, :]

plt.plot(transformed_class1[:, 0], transformed_class1[:, 1], 'x', markersize=7, color='blue', alpha=0.5, label='class1')
plt.plot(transformed_class2[:, 0], transformed_class2[:, 1], '^', markersize=7, color='red', alpha=0.5, label='class2')

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.title('Transformed samples (sklearn PCA)')

plt.show()
