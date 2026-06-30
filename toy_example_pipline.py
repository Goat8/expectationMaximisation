"""
TOY EXAMPLE: EM-based Gene Regulatory Network Inference
Uses the SAME classes as the real DREAM5 E. coli pipeline:
  - ExpectationMaximization      (Level 1: clustering)
  - EMNetworkInference           (Level 2: pairwise edge inference)
  - BayesianNetworkGRN           (Level 3: Friedman-style structure search)

Only the DATA is small here (7 genes, 12 conditions) so every step
is traceable by hand. The algorithms running on it are identical to
the ones that ran on the full 4511-gene E. coli dataset.

Ground truth built in by construction:
  - TF_A regulates Gene_1, Gene_2 (activation)
  - TF_B regulates Gene_3, Gene_4 (repression)
  - Gene_5 has no true regulator (noise)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import networkx as nx
from expectation_maximization import ExpectationMaximization
from EM_network_inference import EMNetworkInference
from bayesian_network import BayesianNetworkGRN

np.random.seed(7)
os.makedirs("toy_images", exist_ok=True)

# =====================================================
# STEP 0: BUILD TOY DATA WITH KNOWN GROUND TRUTH
# =====================================================
n_conditions = 12
genes = ["TF_A", "TF_B", "Gene_1", "Gene_2", "Gene_3", "Gene_4", "Gene_5"]
is_tf = [True, True, False, False, False, False, False]
tf_indices = [0, 1]
tf_gene_ids = ["TF_A", "TF_B"]

TF_A = np.random.normal(5, 1.0, n_conditions)
TF_B = np.random.normal(5, 1.0, n_conditions)

Gene_1 = 0.9 * TF_A + np.random.normal(0, 0.3, n_conditions)
Gene_2 = 0.7 * TF_A + np.random.normal(0, 0.3, n_conditions)

Gene_3 = -0.8 * TF_B + 10 + np.random.normal(0, 0.3, n_conditions)
Gene_4 = -0.6 * TF_B + 10 + np.random.normal(0, 0.3, n_conditions)

Gene_5 = np.random.normal(5, 1.0, n_conditions)

X = np.vstack([TF_A, TF_B, Gene_1, Gene_2, Gene_3, Gene_4, Gene_5])
gene_ids = genes

print("=" * 60)
print("TOY EXPRESSION MATRIX (7 genes x 12 conditions)")
print("=" * 60)
df_X = pd.DataFrame(X, index=genes, columns=[f"C{i+1}" for i in range(n_conditions)])
print(df_X.round(2))

GROUND_TRUTH = {
    ("TF_A", "Gene_1"): "activation",
    ("TF_A", "Gene_2"): "activation",
    ("TF_B", "Gene_3"): "repression",
    ("TF_B", "Gene_4"): "repression",
}
print("\nGround truth regulatory edges (built in by construction):")
for (tf, tgt), kind in GROUND_TRUTH.items():
    print(f"  {tf} -> {tgt}  ({kind})")
print("  Gene_5: no true regulator (noise)")

# =====================================================
# LEVEL 1: EM CLUSTERING  (real ExpectationMaximization class)
# =====================================================
print("\n" + "=" * 60)
print("LEVEL 1: EM CLUSTERING  (using ExpectationMaximization)")
print("=" * 60)

K = 2
em = ExpectationMaximization(num_clusters=K)

# Initialize exactly as main.py does for the real pipeline
em.mixing_coefficients = np.ones(K) / K
init_idx = [0, 5]  # TF_A and Gene_4 as initial centers
em.cluster_means = X[init_idx].copy()
em.cluster_std = np.ones(K) * X.std()

print(f"\nInitial means (gene indices {init_idx}): {[genes[i] for i in init_idx]}")

prev_ll = -np.inf
for iteration in range(30):
    responsibilities = em.expectation(X)
    em.maximization(X, responsibilities)
    ll = em.log_likelihood(X)
    if abs(ll - prev_ll) < 1e-6:
        print(f"Converged at iteration {iteration}")
        break
    prev_ll = ll

clusters = responsibilities.argmax(axis=1)
print(f"\nFinal cluster assignments:")
for g, c, probs in zip(genes, clusters, responsibilities):
    print(f"  {g:8s} -> Cluster {c}  (P={probs.round(3)})")

# =====================================================
# LEVEL 2: PAIRWISE EM EDGE INFERENCE  (real EMNetworkInference class)
# =====================================================
print("\n" + "=" * 60)
print("LEVEL 2: PAIRWISE EM EDGE INFERENCE  (using EMNetworkInference)")
print("=" * 60)

em_net = EMNetworkInference(max_iter=30, tol=1e-6, edge_prior=0.3)

all_l2_results = []
for gene_idx, gname in enumerate(genes):
    if gene_idx in tf_indices:
        continue
    results = em_net.infer_edges_for_gene(
        gene_idx=gene_idx,
        gene_expr=X[gene_idx],
        tf_indices=tf_indices,
        tf_gene_ids=tf_gene_ids,
        X=X
    )
    for r in results:
        r["target"] = gname
        r["is_true_edge"] = (r["tf"], gname) in GROUND_TRUTH
        all_l2_results.append(r)

l2_df = pd.DataFrame(all_l2_results).sort_values("edge_probability", ascending=False)
print(f"\n{'TF':8s} {'Target':8s} {'Edge Prob':>10s} {'Beta':>8s}  {'True Edge?'}")
for _, row in l2_df.iterrows():
    marker = "<-- TRUE" if row["is_true_edge"] else ""
    print(f"{row['tf']:8s} {row['target']:8s} {row['edge_probability']:>10.3f} {row['beta']:>8.3f}  {marker}")

# =====================================================
# LEVEL 3: BAYESIAN NETWORK STRUCTURE LEARNING  (real BayesianNetworkGRN class)
# =====================================================
print("\n" + "=" * 60)
print("LEVEL 3: BAYESIAN NETWORK STRUCTURE LEARNING  (using BayesianNetworkGRN)")
print("=" * 60)

bn = BayesianNetworkGRN(X=X, gene_ids=gene_ids, tf_indices=tf_indices)

non_tf_indices = [i for i in range(len(genes)) if i not in tf_indices]
bn_edges = []
for target_idx in non_tf_indices:
    print(f"\n  Target: {genes[target_idx]}")
    parents, score = bn.greedy_hill_climbing(
        target_idx=target_idx,
        max_parents=2,
        verbose=True
    )
    for p in parents:
        bn_edges.append({
            "tf": genes[p],
            "target": genes[target_idx],
            "bic_score": score
        })

bn_df = pd.DataFrame(bn_edges)

# =====================================================
# VALIDATION AGAINST GROUND TRUTH
# =====================================================
print("\n" + "=" * 60)
print("VALIDATION AGAINST GROUND TRUTH")
print("=" * 60)

l2_top4 = l2_df.head(4)
l2_hits = sum(1 for _, row in l2_top4.iterrows() if row["is_true_edge"])
print(f"\nLevel 2 (Pairwise EM, via EMNetworkInference) top 4 predictions: {l2_hits}/4 correct")

bn_hits = sum(
    1 for _, row in bn_df.iterrows()
    if (row["tf"], row["target"]) in GROUND_TRUTH
)
print(f"Level 3 (Bayesian Network, via BayesianNetworkGRN) all predictions: {bn_hits}/{len(bn_df)} correct")

# =====================================================
# VISUALIZATION
# =====================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].set_title("Toy Expression Data\n(TF_A activates Gene_1/2, TF_B represses Gene_3/4)")
for i, g in enumerate(genes):
    style = "-o" if g in ["TF_A", "TF_B"] else "--s"
    axes[0].plot(range(1, n_conditions + 1), X[i], style, label=g, alpha=0.8)
axes[0].set_xlabel("Condition")
axes[0].set_ylabel("Expression")
axes[0].legend(fontsize=8, ncol=2)
axes[0].grid(alpha=0.3)

colors = ["#534AB7" if g in ["TF_A", "TF_B"] else "#1D9E75" for g in genes]
axes[1].bar(genes, clusters + 1, color=colors)
axes[1].set_title("Level 1: EM Cluster Assignment\n(via ExpectationMaximization class)")
axes[1].set_ylabel("Cluster ID")
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("toy_images/toy_overview.png", dpi=200, bbox_inches="tight")
plt.close()
print("\nSaved: toy_images/toy_overview.png")


G_truth = nx.DiGraph()
for (tf, tgt), kind in GROUND_TRUTH.items():
    G_truth.add_edge(tf, tgt)

# --- Level 2 inferred network (top 4 edges by probability, regardless of correctness) ---
G_l2 = nx.DiGraph()
for _, row in l2_df.head(4).iterrows():
    G_l2.add_edge(row["tf"], row["target"])

# --- Level 3 inferred network (everything BayesianNetworkGRN actually selected) ---
G_l3 = nx.DiGraph()
for _, row in bn_df.iterrows():
    G_l3.add_edge(row["tf"], row["target"])

def draw_network(G, title, filename, true_edges=GROUND_TRUTH):
    plt.figure(figsize=(6.5, 5.5))
    pos = nx.spring_layout(G, seed=3, k=1.3) if len(G.edges) > 0 else {g: (0, 0) for g in genes}
    node_colors = ["#534AB7" if n in ["TF_A", "TF_B"] else "#1D9E75" for n in G.nodes()]

    # Color edges: green if matches ground truth, red if a false positive
    edge_colors = []
    for u, v in G.edges():
        edge_colors.append("#1D9E75" if (u, v) in true_edges else "#D85A30")

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2600)
    nx.draw_networkx_labels(G, pos, font_color="white", font_weight="bold", font_size=11)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True, arrowsize=22, width=1.8)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"toy_images/{filename}", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: toy_images/{filename}")

draw_network(G_truth, "Ground Truth Network\n(built into toy data)", "ground_truth_network.png")
draw_network(G_l2, "Level 2 Inferred Network\n(top 4 edges by probability)\ngreen = correct, orange = false positive", "level2_inferred_network.png")
draw_network(G_l3, "Level 3 Inferred Network\n(Bayesian Network, all selected edges)\ngreen = correct, orange = false positive", "level3_inferred_network.png")

print("\nDONE. Toy example complete — all three levels used the actual production classes.")