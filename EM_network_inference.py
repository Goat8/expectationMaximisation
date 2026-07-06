import numpy as np
import pandas as pd
from scipy import stats


class EMNetworkInference:

    def __init__(self, max_iter=50, tol=1e-4, edge_prior=0.1):
        self.max_iter = max_iter
        self.tol = tol
        self.pi = edge_prior

    def _log_normal(self, y, mu, sigma):
        """
        Log-likelihood of y under N(mu, sigma^2).
        Summed across all T conditions — returns a scalar.

        FIX: previously this was computed per-condition
        and then averaged AFTER taking the ratio.
        Correct approach: sum log-likelihoods first,
        apply Bayes rule once on the totals.
        """
        T = len(y)
        return (
            -0.5 * T * np.log(2 * np.pi * sigma ** 2)
            - 0.5 * np.sum((y - mu) ** 2) / (sigma ** 2)
        )

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

        FIX 1: sum log-likelihoods across all conditions
        THEN apply Bayes rule once — not per-condition ratio
        averaged afterward.

        FIX 2: log-sum-exp over just two terms to avoid
        underflow when likelihoods are very small.

        Returns: edge_prob scalar in [0, 1]
        """
        predicted = beta * tf_expr

        # Sum log-likelihood across all conditions
        log_lik_reg  = (
            np.log(self.pi + 1e-300)
            + self._log_normal(gene_expr, predicted, sigma_reg)
        )
        log_lik_base = (
            np.log(1 - self.pi + 1e-300)
            + self._log_normal(gene_expr, mu_base, sigma_base)
        )

        # Log-sum-exp over two terms
        log_max = max(log_lik_reg, log_lik_base)
        log_lik_reg  -= log_max
        log_lik_base -= log_max

        edge_prob = np.exp(log_lik_reg) / (
            np.exp(log_lik_reg) + np.exp(log_lik_base)
        )

        return float(edge_prob)

    def _m_step(self, tf_expr, gene_expr, edge_prob, sigma_base):
        """
        M-step for one TF-gene pair.

        FIX 1 (beta): edge_prob is a scalar — it cancels from
        numerator and denominator of weighted OLS, so beta
        is just unweighted OLS. Previous code kept edge_prob
        in, which was mathematically redundant but obscured
        the derivation and breaks if you ever extend to
        per-condition weights.

        FIX 2 (sigma_reg): soft mixture update.
        sigma_reg is a blend of regulated residual variance
        and baseline variance, weighted by edge_prob.
        Previous code used sqrt(mean(w * residuals^2))
        which is neither the regulated nor baseline MLE —
        it was an incorrect hybrid.

        Returns: beta, sigma_reg
        """
        # Unweighted OLS for beta — edge_prob cancels
        denom = np.sum(tf_expr ** 2)
        if denom < 1e-10:
            beta = 0.0
        else:
            beta = np.sum(tf_expr * gene_expr) / denom

        # Soft mixture sigma update
        residuals = gene_expr - beta * tf_expr
        var_reg  = np.mean(residuals ** 2)
        var_base = sigma_base ** 2

        # Blend: if edge_prob=1 → pure regulated residual
        #        if edge_prob=0 → pure baseline noise
        var_new  = edge_prob * var_reg + (1 - edge_prob) * var_base
        sigma_floor = 0.5 * sigma_base  # cannot shrink below half baseline noise
        sigma_reg = max(np.sqrt(var_new), sigma_floor) + 1e-10

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
        For one target gene, run EM over all TFs.
        Returns list of dicts: tf_id, edge_probability, beta.
        """
        mu_base    = gene_expr.mean()
        sigma_base = gene_expr.std() + 1e-10

        results = []

        for tf_idx, tf_id in zip(tf_indices, tf_gene_ids):

            if tf_idx == gene_idx:
                continue

            tf_expr = X[tf_idx]

            # Initialize beta from correlation, not 1.0
            corr = np.corrcoef(tf_expr, gene_expr)[0, 1]
            beta      = corr if np.isfinite(corr) else 0.0
            sigma_reg = sigma_base
            edge_prob = self.pi

            prev_edge_prob = -1.0

            for _ in range(self.max_iter):

                # E-step
                edge_prob = self._compute_edge_posterior(
                    tf_expr, gene_expr,
                    beta, sigma_reg,
                    mu_base, sigma_base
                )

                # M-step
                beta, sigma_reg = self._m_step(
                    tf_expr, gene_expr,
                    edge_prob, sigma_base
                )

                # FIX 3: convergence on relative change in
                # edge_prob — previous tol=1e-4 on raw value
                # is fine here since edge_prob is in [0,1],
                # but guard against first-iteration comparison
                if abs(edge_prob - prev_edge_prob) < self.tol:
                    break

                prev_edge_prob = edge_prob

            results.append({
                'tf':               tf_id,
                'edge_probability': edge_prob,
                'beta':             beta,
            })

        return results