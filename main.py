import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from expectation_maximization import ExpectationMaximization
from data import generate_synthetic_gene_expression_data

X = generate_synthetic_gene_expression_data()
# =====================================================
# CREATE EM MODEL
# =====================================================
num_clusters = 2

em = ExpectationMaximization(num_clusters)

# =====================================================
# INITIALIZE PARAMETERS
# =====================================================
em.mixing_coefficients = np.array([0.5, 0.5])

em.cluster_means = np.random.randn(
    num_clusters,
    X.shape[1]
)

em.cluster_std = np.ones(num_clusters)

# =====================================================
# HISTORY STORAGE
# =====================================================
log_likelihood_history = []

cluster_mean_history = []

sample_index = 0

responsibility_history = []

# =====================================================
# EM LOOP
# =====================================================
previous_log_likelihood = -np.inf

for iteration in range(100):

    responsibilities = em.expectation(X)

    responsibility_history.append(
        responsibilities[sample_index, 0]
    )

    em.maximization(
        X,
        responsibilities
    )

    cluster_mean_history.append(
        em.cluster_means.copy()
    )

    current_log_likelihood = em.log_likelihood(X)

    log_likelihood_history.append(
        current_log_likelihood
    )

    print(
        f"Iteration {iteration:02d} "
        f"LogLikelihood = "
        f"{current_log_likelihood:.4f}"
    )

    if abs(
        current_log_likelihood
        - previous_log_likelihood
    ) < 1e-6:

        print(
            f"\nConverged at iteration {iteration}"
        )

        break

    previous_log_likelihood = (
        current_log_likelihood
    )

# =====================================================
# FINAL CLUSTER ASSIGNMENTS
# =====================================================
clusters = responsibilities.argmax(axis=1)

print("\nFirst 10 assignments:")

print(clusters[:10])

# =====================================================
# VISUALIZATION 1
# LOG LIKELIHOOD
# =====================================================
plt.figure(figsize=(6,4))

plt.plot(
    log_likelihood_history,
    marker="o"
)

plt.title("EM Convergence")
plt.xlabel("Iteration")
plt.ylabel("Log Likelihood")
plt.grid(True)
plt.show()

# =====================================================
# VISUALIZATION 2
# RESPONSIBILITY EVOLUTION
# =====================================================
plt.figure(figsize=(6,4))

plt.plot(
    responsibility_history,
    marker="o"
)

plt.title(
    f"Responsibility Evolution\nSample {sample_index}"
)

plt.xlabel("Iteration")
plt.ylabel("P(Cluster 0)")
plt.grid(True)
plt.show()

# =====================================================
# VISUALIZATION 3
# CLUSTER MEAN EVOLUTION
# =====================================================
plt.figure(figsize=(7,4))

for cluster in range(num_clusters):

    plt.plot(
        [
            means[cluster, 0]
            for means in cluster_mean_history
        ],
        label=f"Cluster {cluster}"
    )

plt.title(
    "Evolution of Cluster Means\n(First Feature)"
)

plt.xlabel("Iteration")
plt.ylabel("Mean Value")
plt.legend()
plt.grid(True)
plt.show()

# =====================================================
# PCA VISUALIZATION
# =====================================================
pca = PCA(n_components=2)

X_2d = pca.fit_transform(X)

plt.figure(figsize=(7,6))

plt.scatter(
    X_2d[:,0],
    X_2d[:,1]
)

plt.title("Before EM")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

plt.figure(figsize=(7,6))

plt.scatter(
    X_2d[:,0],
    X_2d[:,1],
    c=clusters,
    s=50
)

plt.title("Gene Clusters After EM")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Cluster")
plt.show()