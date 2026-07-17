"""
Spike-and-slab EM for gene regulatory network inference — PERFORMANCE-FIXED VERSION.

Identical math to the original em_single_target. The only change: instead of
rebuilding each leave-one-out residual from scratch with a fresh O(p) Python sum
(making every step O(p^2)), we compute the full prediction ONCE per step and
subtract/add back each TF's own contribution in O(1) per TF. This turns each
E-step and M-step from O(p^2) into O(p), which at p=685 is roughly a 685x
reduction in the redundant work.

Drop-in replacement: same function signature, same return values.
"""

import numpy as np


def standardize(M):
    """Center to mean 0 and scale to unit variance, column-wise."""
    M = M - M.mean(axis=0, keepdims=True)
    s = M.std(axis=0, keepdims=True)
    s[s == 0] = 1.0
    return M / s


def em_single_target(y, X, tau2=5.0, max_iter=100, tol=1e-6, sigma2_floor=1e-4):
    """
    Run spike-and-slab EM for ONE target gene. Math identical to the original;
    only the residual bookkeeping is faster.

    Parameters
    ----------
    y : (n,) target expression, already standardized
    X : (n, p) TF expression matrix, already standardized
    tau2 : slab prior variance

    Returns
    -------
    gamma : (p,) edge posterior per TF
    beta  : (p,) regulatory strength per TF
    sigma2: residual noise variance
    """
    n, p = X.shape
    beta = np.zeros(p)
    sigma2 = float(np.var(y))
    pi = 0.5
    gamma = np.full(p, 0.5)
    prev_ll = -np.inf

    # Precompute column norms once -- reused every iteration, never changes
    xtx_all = np.einsum('ij,ij->j', X, X)  # (p,) -- x_j^T x_j for every j at once

    for _ in range(max_iter):
        # ---------- E-STEP: marginalized edge posterior ----------
        # Full current prediction, computed ONCE (O(n*p)) instead of p times (O(p^2))
        pred_full = X @ (gamma * beta)  # (n,)

        new_gamma = np.zeros(p)
        for j in range(p):
            # leave-one-out residual: subtract full prediction, add back j's own term
            r = y - pred_full + gamma[j] * beta[j] * X[:, j]

            xj = X[:, j]
            xtx = xtx_all[j]
            xtr = xj @ r

            a = tau2 / (sigma2 * (sigma2 + tau2 * xtx))
            quad_null = (r @ r) / sigma2
            quad_slab = quad_null - a * xtr ** 2
            logdet_ratio = np.log1p(tau2 * xtx / sigma2)

            log_slab = np.log(pi)    - 0.5 * quad_slab - 0.5 * logdet_ratio
            log_null = np.log1p(-pi) - 0.5 * quad_null
            m = max(log_slab, log_null)
            new_gamma[j] = np.exp(log_slab - m) / (
                np.exp(log_slab - m) + np.exp(log_null - m)
            )
        gamma = new_gamma

        # ---------- M-STEP: strengths, noise, prior ----------
        # Recompute pred_full with the just-updated gamma (beta not yet updated this round)
        pred_full = X @ (gamma * beta)
        for j in range(p):
            r = y - pred_full + gamma[j] * beta[j] * X[:, j]
            beta_j_new = (X[:, j] @ r) / (xtx_all[j] + sigma2 / tau2)
            # keep pred_full consistent as beta[j] changes, so the NEXT j in this
            # same M-step loop sees an up-to-date prediction (matches original's
            # sequential-update behaviour)
            pred_full = pred_full - gamma[j] * beta[j] * X[:, j] + gamma[j] * beta_j_new * X[:, j]
            beta[j] = beta_j_new

        resid = y - X @ (gamma * beta)
        sigma2 = max(float(np.var(resid)), sigma2_floor)
        pi = float(gamma.mean())

        ll = -0.5 * (resid @ resid) / sigma2 - 0.5 * n * np.log(sigma2)
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    return gamma, beta, sigma2