import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from expectation_maximization import ExpectationMaximization
from data import generate_synthetic_gene_expression_data
from data import load_dream5_ecoli
import os

os.makedirs("images", exist_ok=True)

USE_REAL_DATA = True
num_clusters = 5

if USE_REAL_DATA:
    X, gene_names, gene_ids, is_tf = load_dream5_ecoli(
        expression_path='data/network_ecoli/net3_expression_data.tsv',
        gene_ids_path='data/network_ecoli/net3_gene_ids.tsv',
        tf_path='data/network_ecoli/net3_transcription_factors.tsv'
    )
    # Standardize: zero mean, unit variance per gene
    scaler = StandardScaler()
    X = scaler.fit_transform(X.T).T
    num_clusters = 5  # more clusters for real data
else:
    X = generate_synthetic_gene_expression_data()
    gene_names = [f'Gene_{i}' for i in range(100)]
    is_tf = [False] * 100
    num_clusters = 2
#X = generate_synthetic_gene_expression_data()
# =====================================================
# CREATE EM MODEL
# =====================================================

em = ExpectationMaximization(num_clusters)

# =====================================================
# INITIALIZE PARAMETERS
# =====================================================
# em.mixing_coefficients = np.array([0.5, 0.5])
em.mixing_coefficients = np.ones(num_clusters) / num_clusters

# em.cluster_means = np.random.randn(
#     num_clusters,
#     X.shape[1]
# )
random_indices = np.random.choice(X.shape[0], num_clusters, replace=False)
em.cluster_means = X[random_indices].copy()
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


plt.savefig(
    "images/log_likelihood.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

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

plt.savefig(
    "images/responsibility_evolution.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()
plt.close()
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

plt.savefig(
    "images/cluster_mean_evolution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()
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
plt.savefig(
    "images/pca_before_em.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

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
plt.savefig(
    "images/pca_after_em.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

# =====================================================
# BIOLOGICAL INTERPRETATION
# TF ENRICHMENT PER CLUSTER
# =====================================================
print("\n=== FINAL CLUSTER SUMMARY ===")
is_tf_array = np.array(is_tf)

for k in range(num_clusters):
    cluster_mask = clusters == k
    cluster_size = cluster_mask.sum()
    tf_in_cluster = (cluster_mask & is_tf_array).sum()
    tf_enrichment = tf_in_cluster / cluster_size * 100
    
    # top 5 gene names in this cluster
    cluster_gene_names = [
        gene_names[i] 
        for i in range(len(gene_names)) 
        if cluster_mask[i]
    ][:5]
    
    print(f"Cluster {k}: {cluster_size} genes | "
          f"TFs: {tf_in_cluster} ({tf_enrichment:.1f}%) | "
          f"samples: {cluster_gene_names}")