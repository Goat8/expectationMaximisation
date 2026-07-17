"""
Wire TCGA-BRCA expression data + TRRUST TF list into the slab-spike EM pipeline.

Inputs:
  brca_tpm_matrix.csv       - log2(TPM+1) expression matrix (genes x patients)
  trrust_rawdata.human.tsv  - TRRUST v2 human TF-target interactions

Output:
  brca_grn_results.csv      - inferred edges: TF, target, gamma, beta, direction
  brca_adjacency.csv        - binary adjacency matrix (TFs x targets)
"""

import numpy as np
import pandas as pd
from slab_spike_em_updated import standardize, em_single_target


def load_expression(path="brca_tpm_matrix.csv"):
    """Load expression matrix (genes x patients) -> transpose to (patients x genes)."""
    df = pd.read_csv(path, index_col=0)
    print(f"Expression matrix loaded: {df.shape[0]} genes x {df.shape[1]} patients")
    return df


def load_trrust(path="trrust_rawdata.human.tsv"):
    """Load TRRUST and return TF set, target set, and interaction DataFrame."""
    tr = pd.read_csv(path, sep="\t", header=None,
                     names=["TF", "target", "direction", "pmid"])
    tfs = set(tr["TF"].unique())
    targets = set(tr["target"].unique())
    print(f"TRRUST loaded: {len(tr)} interactions, {len(tfs)} TFs, {len(targets)} targets")
    return tr, tfs, targets


def prepare_matrices(expr_df, trrust_df, trrust_tfs):
    """
    Split expression into TF matrix and target matrix.
    Only keep TFs and targets present in the expression data.
    Only keep targets that have at least one known TF in TRRUST.
    """
    expr_genes = set(expr_df.index)

    # TFs present in expression data
    tf_genes = sorted(trrust_tfs & expr_genes)

    # Targets: genes that appear as targets in TRRUST and are in expression data
    # Exclude TFs from target list to avoid self-loops
    trrust_targets = set(trrust_df["target"].unique())
    target_genes = sorted((trrust_targets & expr_genes) - set(tf_genes))

    print(f"TFs in expression data: {len(tf_genes)}")
    print(f"Targets in expression data: {len(target_genes)}")

    # Transpose: (patients x genes)
    X_tf = expr_df.loc[tf_genes].values.T       # (n_patients, n_tfs)
    Y_targets = expr_df.loc[target_genes].values.T  # (n_patients, n_targets)

    return X_tf, Y_targets, tf_genes, target_genes


def run_em_all_targets(X_tf, Y_targets, tf_genes, target_genes, tau2=5.0,
                       threshold=0.5, verbose_every=100):
    """
    Run slab-spike EM for each target gene against all TFs.

    Returns gamma and beta matrices of shape (n_targets, n_tfs).
    """
    n_patients, n_tfs = X_tf.shape
    n_targets = Y_targets.shape[1]

    print(f"\nRunning EM: {n_targets} targets x {n_tfs} TFs, {n_patients} patients")
    print(f"tau2={tau2}, threshold={threshold}\n")

    X_std = standardize(X_tf)

    gamma_mat = np.zeros((n_targets, n_tfs))
    beta_mat = np.zeros((n_targets, n_tfs))

    for i in range(n_targets):
        y = standardize(Y_targets[:, [i]]).ravel()
        g, b, s2 = em_single_target(y, X_std, tau2=tau2)
        gamma_mat[i] = g
        beta_mat[i] = b

        if (i + 1) % verbose_every == 0 or i == 0:
            n_edges = (g > threshold).sum()
            print(f"  Target {i+1}/{n_targets}: {target_genes[i]:>10s} | "
                  f"edges={n_edges}, sigma2={s2:.4f}")

    print(f"\nEM complete.")
    return gamma_mat, beta_mat


def save_results(gamma_mat, beta_mat, tf_genes, target_genes, trrust_df,
                 threshold=0.5):
    """Save edge list and adjacency matrix."""
    # Build direction lookup from TRRUST
    direction_map = {}
    for _, row in trrust_df.iterrows():
        direction_map[(row["TF"], row["target"])] = row["direction"]

    # Edge list: only edges above threshold
    edges = []
    for i, tgt in enumerate(target_genes):
        for j, tf in enumerate(tf_genes):
            if gamma_mat[i, j] > threshold:
                edges.append({
                    "TF": tf,
                    "target": tgt,
                    "gamma": gamma_mat[i, j],
                    "beta": beta_mat[i, j],
                    "trrust_direction": direction_map.get((tf, tgt), "Novel"),
                })

    edge_df = pd.DataFrame(edges)
    edge_df = edge_df.sort_values("gamma", ascending=False)
    edge_df.to_csv("brca_grn_results.csv", index=False)

    # Adjacency matrix
    adj = (gamma_mat > threshold).astype(int)
    adj_df = pd.DataFrame(adj, index=target_genes, columns=tf_genes)
    adj_df.to_csv("brca_adjacency.csv")

    n_edges = adj.sum()
    n_possible = gamma_mat.shape[0] * gamma_mat.shape[1]
    print(f"Edges inferred: {n_edges} / {n_possible} possible "
          f"({100*n_edges/n_possible:.2f}%)")
    print(f"Results saved: brca_grn_results.csv, brca_adjacency.csv")

    return edge_df


def main():
    # Load data
    expr_df = load_expression()
    trrust_df, trrust_tfs, _ = load_trrust()

    # Prepare matrices
    X_tf, Y_targets, tf_genes, target_genes = prepare_matrices(
        expr_df, trrust_df, trrust_tfs
    )

    # Run EM
    gamma_mat, beta_mat = run_em_all_targets(
        X_tf, Y_targets, tf_genes, target_genes, tau2=5.0,
        verbose_every=10
    )

    # Save
    edge_df = save_results(gamma_mat, beta_mat, tf_genes, target_genes, trrust_df)

    # Summary
    print(f"\nTop 20 inferred edges:")
    print(edge_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
