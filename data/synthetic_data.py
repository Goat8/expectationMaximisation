import numpy as np


def generate_synthetic_gene_expression_data(
    num_samples_per_cluster=50,
    num_genes=10,
    random_seed=42
):
    np.random.seed(random_seed)

    return np.vstack([
        np.random.randn(
            num_samples_per_cluster,
            num_genes
        ) + 2,

        np.random.randn(
            num_samples_per_cluster,
            num_genes
        ) - 2
    ])