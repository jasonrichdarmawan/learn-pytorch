# %%
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. Create some sample 2D data
# Let's imagine data points that are somewhat correlated
rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
# X now has 200 samples and 2 features

# 2. Instantiate PCA
# We want to reduce the dimensionality to 1 component
pca = PCA(n_components=1)

# 3. Fit PCA to the data and transform the data
# fit_transform() first learns the principal components from X (fit)
# and then applies the dimensionality reduction to X (transform)
X_pca = pca.fit_transform(X)

# X_pca will now have 200 samples and 1 feature

# Let's see what happened
print("Original shape: ", X.shape)
print("Transformed shape:", X_pca.shape)

# The pca object now contains information about the transformation
print("Explained variance ratio:", pca.explained_variance_ratio_) # How much variance is captured by the new component
print("Principal components (eigenvectors):\n", pca.components_) # The direction(s) of maximum variance

# To understand fit_transform better, you could do it in two steps:
pca_separate = PCA(n_components=1)
pca_separate.fit(X) # Step 1: Learn the transformation
X_pca_separate = pca_separate.transform(X) # Step 2: Apply the transformation
print("Transformed shape (separate steps):", X_pca_separate.shape)
print("Are the results the same?", np.allclose(X_pca, X_pca_separate))


# 4. Visualize (optional, but helpful for 2D -> 1D)
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.7, label='Original data')

# To plot the transformed data back in the original space, we can use inverse_transform
X_reconstructed = pca.inverse_transform(X_pca)
plt.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], alpha=0.7, label='PCA reconstructed (1D projection)')

# Plot the principal component direction
# The component vector starts at the mean of the data
mean_x = np.mean(X[:, 0])
mean_y = np.mean(X[:, 1])
component = pca.components_[0]
# Scale the component for visualization
arrow_length = 3 * np.sqrt(pca.explained_variance_[0]) # Length proportional to explained variance
plt.arrow(mean_x, mean_y, arrow_length * component[0], arrow_length * component[1],
          head_width=0.1, head_length=0.2, fc='red', ec='red', label='Principal Component 1')


plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("PCA: 2D to 1D")
plt.legend()
plt.axis('equal') # Important for visualizing PCA directions correctly
plt.show()

print("\nOriginal first 5 samples:\n", X[:5])
print("\nTransformed first 5 samples (1D):\n", X_pca[:5])
print("\nReconstructed first 5 samples from 1D PCA:\n", X_reconstructed[:5])
