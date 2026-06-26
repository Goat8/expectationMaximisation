import numpy as np
import pandas as pd
from scipy import stats

class EMNetworkInference:
    def __init__(
        self,
        max_iter=50,
        tol=1e-4,
        edge_prior=0.1
    ):
        # P(theta) Prior probability that any TF-gene edge exists
        # 0.1 means we assume 10% of TF-gene pairs are real
        self.max_iter = max_iter
        self.tol = tol
        self.pi = edge_prior

    def _compute_edge_posterior(
        self,
        tf_expr,
        gene_expr,
        beta,
        sigma_reg,
        mu_base,
        sigma_base
    ):
        """
        E-step for one TF-gene pair.
        
        Computes P(Z=1 | X) — probability that
        the regulatory edge exists.
        
        Parameters:
            tf_expr: TF expression across conditions (805,)
            gene_expr: target gene expression (805,)
            beta: current regulatory strength estimate
            sigma_reg: noise under regulation
            mu_base: baseline mean expression of gene
            sigma_base: baseline std of gene
        
        Returns:
            edge_prob: scalar, P(edge exists | data)
        """
        # Likelihood if edge EXISTS:
        # gene expression explained by TF
        predicted = beta * tf_expr
        regulated_likelihood = stats.norm.pdf(
            gene_expr,
            loc=predicted,
            scale=sigma_reg
        )
        
        # Likelihood if edge DOES NOT EXIST:
        # gene expression from baseline distribution
        baseline_likelihood = stats.norm.pdf(
            gene_expr,
            loc=mu_base,
            scale=sigma_base
        )
        
        # Posterior via Bayes rule
        numerator = self.pi * regulated_likelihood
        denominator = (
            self.pi * regulated_likelihood
            + (1 - self.pi) * baseline_likelihood
        )
        
        # Average across conditions for stability
        edge_prob = np.mean(
            numerator / (denominator + 1e-300)
        )
        
        return edge_prob

    def _m_step(
        self,
        tf_expr,
        gene_expr,
        edge_prob
    ):
        """
        M-step for one TF-gene pair.
        
        Updates regulatory strength beta and noise sigma
        using weighted least squares.
        
        Parameters:
            tf_expr: TF expression (805,)
            gene_expr: target gene expression (805,)
            edge_prob: current posterior P(Z=1)
        
        Returns:
            beta: updated regulatory strength
            sigma_reg: updated noise estimate
        """
        # Weighted least squares:
        # weight = edge_prob (how much we believe
        # this edge exists)
        w = edge_prob
        
        # MLE for beta: weighted covariance / variance
        beta = (
            np.sum(w * tf_expr * gene_expr)
            / (np.sum(w * tf_expr ** 2) + 1e-10)
        )
        
        # MLE for sigma: weighted residual std
        residuals = gene_expr - beta * tf_expr
        sigma_reg = np.sqrt(
            np.mean(w * residuals ** 2) + 1e-10
        )
        
        return beta, sigma_reg

    def infer_edges_for_gene(
        self,
        gene_idx,
        gene_expr,
        tf_indices,
        tf_gene_ids,
        X
    ):
        """
        For one target gene, run EM over all TFs
        to find which TFs most likely regulate it.
        
        Returns:
            results: list of (tf_id, edge_probability, beta)
        """
        # Baseline distribution of this gene
        mu_base = gene_expr.mean()
        sigma_base = gene_expr.std() + 1e-10
        
        results = []
        
        for tf_idx, tf_id in zip(tf_indices, tf_gene_ids):
            
            tf_expr = X[tf_idx]
            
            # Skip self-regulation
            if tf_idx == gene_idx:
                continue
            
            # Initialize parameters
            beta = np.corrcoef(
                tf_expr, gene_expr
            )[0, 1]
            sigma_reg = sigma_base
            edge_prob = self.pi
            
            # EM loop for this TF-gene pair
            prev_edge_prob = 0
            
            for iteration in range(self.max_iter):
                
                # E-step: compute edge posterior
                edge_prob = self._compute_edge_posterior(
                    tf_expr,
                    gene_expr,
                    beta,
                    sigma_reg,
                    mu_base,
                    sigma_base
                )
                
                # M-step: update parameters
                beta, sigma_reg = self._m_step(
                    tf_expr,
                    gene_expr,
                    edge_prob
                )
                
                # Check convergence
                if abs(edge_prob - prev_edge_prob) < self.tol:
                    break
                    
                prev_edge_prob = edge_prob
            
            results.append({
                'tf': tf_id,
                'edge_probability': edge_prob,
                'beta': beta
            })
        
        return results