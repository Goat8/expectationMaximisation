import numpy as np
from scipy import stats

class BayesianNetworkGRN:
    def __init__(self, X, gene_ids, tf_indices):
        self.X = X          # (n_genes, n_conditions)
        self.gene_ids = gene_ids
        self.tf_indices = tf_indices
        self.n_genes = X.shape[0]
        self.n_conditions = X.shape[1]
        
    def compute_bic_score(self, target_idx, parent_indices):
        """
        Compute BIC score for one gene given its parents.
        Higher is better.
        
        target_idx: index of target gene
        parent_indices: list of TF indices that regulate it
        """
        y = self.X[target_idx]      # target gene expression
        n = self.n_conditions        # number of observations
        
        if len(parent_indices) == 0:
            # No parents: baseline model
            mu = y.mean()
            sigma = y.std() + 1e-10
            log_lik = stats.norm.logpdf(y, mu, sigma).sum()
            d = 2  # mean and sigma
        else:
            # With parents: linear regression
            parent_expr = self.X[parent_indices]  # (n_parents, n_conditions)
            
            # Add intercept
            A = np.vstack([
                parent_expr,
                np.ones(n)
            ]).T  # (n_conditions, n_parents + 1)
            
            # OLS: beta = (A^T A)^{-1} A^T y
            try:
                beta = np.linalg.lstsq(A, y, rcond=None)[0]
                predicted = A @ beta
                residuals = y - predicted
                sigma = residuals.std() + 1e-10
                log_lik = stats.norm.logpdf(
                    residuals, 0, sigma
                ).sum()
                d = len(parent_indices) + 2  # betas + intercept + sigma
            except:
                return -np.inf
        
        # BIC penalty
        bic = log_lik - (d / 2) * np.log(n)
        return bic
    
    def greedy_hill_climbing(
        self,
        target_idx,
        max_parents=3,
        verbose=False
    ):
        """
        For one target gene, find the best set of TF regulators
        using greedy hill climbing on BIC score.
        
        Starts with no parents, adds one TF at a time
        as long as BIC improves.
        """
        current_parents = []
        current_score = self.compute_bic_score(
            target_idx, []
        )
        
        for step in range(max_parents):
            best_new_parent = None
            best_new_score = current_score
            
            # Try adding each TF not already a parent
            for tf_idx in self.tf_indices:
                if tf_idx == target_idx:
                    continue
                if tf_idx in current_parents:
                    continue
                    
                candidate_parents = current_parents + [tf_idx]
                score = self.compute_bic_score(
                    target_idx,
                    candidate_parents
                )
                
                if score > best_new_score:
                    best_new_score = score
                    best_new_parent = tf_idx
            
            if best_new_parent is None: # not improving gotta stop
                break
                
            current_parents.append(best_new_parent)
            current_score = best_new_score
            
            if verbose:
                print(
                    f"  Added TF {self.gene_ids[best_new_parent]}, "
                    f"BIC={current_score:.2f}, "
                    f"parents={len(current_parents)}"
                )
        
        return current_parents, current_score
    
    def infer_network(
        self,
        target_genes=100,
        max_parents=3,
        verbose=True
    ):
        """
        running greedy hill climbing for each target gene.
        Returns predicted edges as a list of (TF, target, BIC_score).
        """
        edges = []
        
        for i in range(min(target_genes, self.n_genes)):
            if verbose and i % 20 == 0:
                print(f"Processing gene {i}/{target_genes}")
                
            parents, score = self.greedy_hill_climbing(
                target_idx=i,
                max_parents=max_parents
            )
            
            for parent_idx in parents:
                edges.append({
                    'tf': self.gene_ids[parent_idx],
                    'target': self.gene_ids[i],
                    'bic_score': score,
                    'n_parents': len(parents)
                })
        
        return edges