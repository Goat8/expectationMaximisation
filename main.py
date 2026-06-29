import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from expectation_maximization import ExpectationMaximization
from data import generate_synthetic_gene_expression_data
from data import load_dream5_ecoli
from data import load_gold_standard
from data import load_gold_standard
from bayesian_network import BayesianNetworkGRN
from EM_network_inference import EMNetworkInference
import pandas as pd
import networkx as nx
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
    
# =====================================================
# GOLD STANDARD VALIDATION
# =====================================================
#DREAM5_NetworkInference_GoldStandard_Network3 - E. coli.tsv
gs = load_gold_standard(
    'data/network_ecoli/test/DREAM5_NetworkInference_GoldStandard_Network3 - E. coli.tsv'
)

# Map gene IDs to cluster assignments
gene_to_cluster = dict(zip(gene_ids, clusters))

# For each known TF-target pair, check if same cluster
same_cluster = 0
total_pairs = 0

for _, row in gs.iterrows():
    tf = row['tf']
    target = row['target']
    if tf in gene_to_cluster and target in gene_to_cluster:
        total_pairs += 1
        if gene_to_cluster[tf] == gene_to_cluster[target]:
            same_cluster += 1

co_cluster_rate = same_cluster / total_pairs * 100

print(f"\n=== GOLD STANDARD VALIDATION ===")
print(f"Known regulatory pairs evaluated: {total_pairs}")
print(f"Pairs in same cluster: {same_cluster}")
print(f"Co-clustering rate: {co_cluster_rate:.2f}%")
print(f"\nBaseline (random, 5 clusters): ~20.00%")
print(f"Your model: {co_cluster_rate:.2f}%")

if co_cluster_rate > 20:
    print("Result: EM clusters are enriched for known regulatory pairs")
else:
    print("Result: clusters do not significantly recover known regulation")



###3
    
##
    
# =====================================================
# NETWORK INFERENCE
# =====================================================

print("\n=== RUNNING EM NETWORK INFERENCE ===")

# Get TF indices and IDs
tf_indices = [
    i for i, t in enumerate(is_tf) if t
]
tf_gene_ids = [
    gene_ids[i] for i in tf_indices
]

print(f"Transcription factors: {len(tf_indices)}")
print(f"Target genes: {len(gene_ids)}")

# Run inference
em_net = EMNetworkInference(
    max_iter=50,
    tol=1e-4,
    edge_prior=0.1
)

all_edges = []

# Run on first 100 genes for speed
# (full run on 4511 genes takes hours)
target_genes_to_run = 100

for gene_idx in range(target_genes_to_run):
    gene_expr = X[gene_idx]
    gene_id = gene_ids[gene_idx]
    
    edges = em_net.infer_edges_for_gene(
        gene_idx=gene_idx,
        gene_expr=gene_expr,
        tf_indices=tf_indices,
        tf_gene_ids=tf_gene_ids,
        X=X
    )
    
    for edge in edges:
        edge['target'] = gene_id
        all_edges.append(edge)
    
    if gene_idx % 10 == 0:
        print(f"Processed {gene_idx}/{target_genes_to_run} genes")

# Convert to dataframe
edges_df = pd.DataFrame(all_edges)

# Sort by edge probability
edges_df = edges_df.sort_values(
    'edge_probability',
    ascending=False
)

print(f"\nTop 20 predicted regulatory edges:")
print(
    edges_df[['tf','target','edge_probability','beta']]
    .head(20)
    .to_string()
)

# Saving predicted network in csv
edges_df.to_csv(
    'data/predicted_network.csv',
    index=False
)
print(f"\nSaved {len(edges_df)} predicted edges")

# =====================================================
# NETWORK INFERENCE VISUALIZATIONS
# =====================================================

# =====================================================
# EDGE PROBABILITY DISTRIBUTION
# =====================================================
plt.figure(figsize=(8, 4))
plt.hist(
    edges_df['edge_probability'],
    bins=50,
    color='#534AB7',
    edgecolor='white',
    linewidth=0.5
)
plt.axvline(
    x=0.5,
    color='red',
    linestyle='--',
    label='Decision threshold (0.5)'
)
plt.xlabel('Edge Probability P(Z=1 | X)')
plt.ylabel('Number of TF-gene pairs')
plt.title('Distribution of Inferred Regulatory Edge Probabilities')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/edge_probability_dist.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# =====================================================
# TOP TFs BY NUMBER OF PREDICTED TARGETS
# =====================================================
high_conf_edges = edges_df[
    edges_df['edge_probability'] > 0.5
]

tf_target_counts = (
    high_conf_edges
    .groupby('tf')['target']
    .count()
    .sort_values(ascending=False)
    .head(15)
)

# Map gene IDs to names for readability
id_to_name = dict(zip(gene_ids, gene_names))

plt.figure(figsize=(10, 5))
bars = plt.bar(
    range(len(tf_target_counts)),
    tf_target_counts.values,
    color='#1D9E75'
)
plt.xticks(
    range(len(tf_target_counts)),
    [id_to_name.get(tf, tf) for tf in tf_target_counts.index],
    rotation=45,
    ha='right'
)
plt.xlabel('Transcription Factor')
plt.ylabel('Number of Predicted Targets')
plt.title('Top 15 TFs by Predicted Regulatory Targets\n(edge probability > 0.5)')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('images/top_tfs.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# =====================================================
# VIZ 3: REGULATORY NETWORK GRAPH
# Top 50 edges only for readability
# =====================================================
top_edges = edges_df.head(50)

G = nx.DiGraph()

for _, row in top_edges.iterrows():
    tf_name = id_to_name.get(row['tf'], row['tf'])
    target_name = id_to_name.get(row['target'], row['target'])
    G.add_edge(
        tf_name,
        target_name,
        weight=row['edge_probability']
    )

# Color TF nodes differently from target nodes
tf_names = set(
    id_to_name.get(tf, tf) for tf in tf_gene_ids
)
node_colors = [
    '#534AB7' if node in tf_names else '#1D9E75'
    for node in G.nodes()
]
node_sizes = [
    800 if node in tf_names else 300
    for node in G.nodes()
]

plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, k=2, seed=42)
nx.draw_networkx_nodes(
    G, pos,
    node_color=node_colors,
    node_size=node_sizes,
    alpha=0.9
)
nx.draw_networkx_edges(
    G, pos,
    edge_color='#AAAAAA',
    arrows=True,
    arrowsize=15,
    width=1.5,
    alpha=0.6
)
nx.draw_networkx_labels(
    G, pos,
    font_size=7,
    font_color='white',
    font_weight='bold'
)

# Legend
legend_elements = [
    Patch(facecolor='#534AB7', label='Transcription Factor'),
    Patch(facecolor='#1D9E75', label='Target Gene')
]
plt.legend(handles=legend_elements, loc='upper left')
plt.title('Inferred Regulatory Network\n(Top 50 edges by EM edge probability)')
plt.axis('off')
plt.tight_layout()
plt.savefig('images/regulatory_network.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# =====================================================
# BETA DISTRIBUTION
# Positive beta = activation, negative = repression
# =====================================================
plt.figure(figsize=(8, 4))
plt.hist(
    high_conf_edges['beta'],
    bins=40,
    color='#378ADD',
    edgecolor='white',
    linewidth=0.5
)
plt.axvline(x=0, color='red', linestyle='--', label='Zero (no effect)')
plt.xlabel('Regulatory Strength (β)')
plt.ylabel('Number of edges')
plt.title('Distribution of Regulatory Strength\n(positive = activation, negative = repression)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/beta_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("\n=== VISUALIZATION SUMMARY ===")
print(f"Total predicted edges: {len(edges_df)}")
print(f"High confidence edges (>0.5): {len(high_conf_edges)}")
print(f"Predicted activations (beta>0): {(high_conf_edges['beta']>0).sum()}")
print(f"Predicted repressions (beta<0): {(high_conf_edges['beta']<0).sum()}")


# =====================================================
# LEVEL 3: BAYESIAN NETWORK STRUCTURE LEARNING
# (Friedman 2004 approach)

print("\n=== FRIEDMAN BAYESIAN NETWORK INFERENCE ===")

bn = BayesianNetworkGRN(
    X=X,
    gene_ids=gene_ids,
    tf_indices=tf_indices
)

bn_edges = bn.infer_network(
    target_genes=100,
    max_parents=3,
    verbose=True
)

bn_edges_df = pd.DataFrame(bn_edges)
bn_edges_df = bn_edges_df.sort_values(
    'bic_score', ascending=False
)

print(f"\nTop 20 predicted edges (Bayesian Network):")
print(
    bn_edges_df[['tf','target','bic_score','n_parents']]
    .head(20)
    .to_string()
)

bn_edges_df.to_csv(
    'data/bayesian_network_edges.csv',
    index=False
)

# =====================================================
# COMPARING: PAIRWISE EM vs BAYESIAN NETWORK
# =====================================================
from data import load_gold_standard

gs = load_gold_standard(
    'data/network_ecoli/test/DREAM5_NetworkInference_GoldStandard_Network3 - E. coli.tsv'
)

gold_pairs = set(
    zip(gs['tf'], gs['target'])
)

def compute_precision(predicted_df, tf_col, target_col, top_k=200):
    top_edges = predicted_df.head(top_k)
    hits = sum(
        1 for _, row in top_edges.iterrows()
        if (row[tf_col], row[target_col]) in gold_pairs
    )
    return hits / top_k * 100

# precision of pairwise EM
em_precision = compute_precision(
    edges_df.sort_values(
        'edge_probability', ascending=False
    ),
    'tf', 'target', top_k=200
)

# checking precision of Bayesian Network
bn_precision = compute_precision(
    bn_edges_df,
    'tf', 'target', top_k=200
)

print(f"\n=== COMPARISON ===")
print(f"Random baseline (top 200): ~{200/152280*100:.2f}%")
print(f"Pairwise EM precision: {em_precision:.2f}%")
print(f"Bayesian Network precision: {bn_precision:.2f}%")



# =====================================================
# BAYESIAN NETWORK VISUALIZATION
# =====================================================

# Build directed graph from Bayesian Network edges
BN_G = nx.DiGraph()

# Use top 50 edges by BIC score
top_bn_edges = bn_edges_df.head(50)

id_to_name = dict(zip(gene_ids, gene_names))

for _, row in top_bn_edges.iterrows():
    tf_name = id_to_name.get(row['tf'], row['tf'])
    target_name = id_to_name.get(
        row['target'], row['target']
    )
    BN_G.add_edge(
        tf_name,
        target_name,
        weight=row['bic_score']
    )

# Color nodes
tf_name_set = set(
    id_to_name.get(gene_ids[i], gene_ids[i])
    for i in tf_indices
)

node_colors = [
    '#534AB7' if node in tf_name_set else '#1D9E75'
    for node in BN_G.nodes()
]

node_sizes = [
    900 if node in tf_name_set else 400
    for node in BN_G.nodes()
]

plt.figure(figsize=(16, 11))
pos = nx.spring_layout(BN_G, k=2.5, seed=42)

nx.draw_networkx_nodes(
    BN_G, pos,
    node_color=node_colors,
    node_size=node_sizes,
    alpha=0.9
)
nx.draw_networkx_edges(
    BN_G, pos,
    edge_color='#AAAAAA',
    arrows=True,
    arrowsize=20,
    arrowstyle='-|>',
    width=1.5,
    alpha=0.7,
    connectionstyle='arc3,rad=0.1'
)
nx.draw_networkx_labels(
    BN_G, pos,
    font_size=7,
    font_color='white',
    font_weight='bold'
)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#534AB7', label='Transcription Factor'),
    Patch(facecolor='#1D9E75', label='Target Gene')
]
plt.legend(
    handles=legend_elements,
    loc='upper left',
    fontsize=10
)

plt.title(
    'Bayesian Network — Inferred Gene Regulatory Network\n'
    '(Top 50 edges by BIC score, Friedman 2004 approach)',
    fontsize=13
)
plt.axis('off')
plt.tight_layout()
plt.savefig(
    'images/bayesian_network.png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()
plt.close()
print("Saved: images/bayesian_network.png")