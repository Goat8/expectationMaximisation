import numpy as np

class ExpectationMaximization:

    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        self.mixing_coefficients = None
        self.cluster_means = None
        self.cluster_var = None  # shape: (num_clusters, num_features) — diagonal covariance

    # --------------------------------------------------
    # FIX 1: DIAGONAL COVARIANCE (was scalar cluster_std)
    # One variance per feature per cluster instead of one
    # scalar per cluster. Removes the spherical assumption
    # that was producing Voronoi pie slices.
    # --------------------------------------------------
    def expectation(self, data):
        num_samples, num_features = data.shape
        log_responsibilities = np.zeros((num_samples, self.num_clusters))

        for cluster in range(self.num_clusters):
            difference = data - self.cluster_means[cluster]  # (N, D)
            var = self.cluster_var[cluster]                   # (D,)

            log_likelihood = (
                -0.5 * np.sum(difference ** 2 / var, axis=1)
                - 0.5 * np.sum(np.log(2 * np.pi * var))
                + np.log(self.mixing_coefficients[cluster])
            )

            log_responsibilities[:, cluster] = log_likelihood

        # Log-sum-exp trick — unchanged, was already correct
        log_max = log_responsibilities.max(axis=1, keepdims=True)
        log_responsibilities -= log_max
        responsibilities = np.exp(log_responsibilities)
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

        return responsibilities

    def maximization(self, data, responsibilities):
        num_samples, num_features = data.shape

        self.mixing_coefficients = (
            responsibilities.sum(axis=0) / num_samples
        )

        self.cluster_means = (
            responsibilities.T @ data
        ) / responsibilities.sum(axis=0)[:, None]

        # FIX 1 (cont): diagonal covariance M-step
        # Per-feature weighted variance instead of pooled scalar
        self.cluster_var = np.zeros((self.num_clusters, num_features))

        for cluster in range(self.num_clusters):
            diff = data - self.cluster_means[cluster]          # (N, D)
            r = responsibilities[:, cluster]                   # (N,)
            self.cluster_var[cluster] = (
                (r[:, None] * diff ** 2).sum(axis=0)
                / r.sum()
            )                                                  # (D,)

        # Guard against variance collapsing to zero
        self.cluster_var = np.maximum(self.cluster_var, 1e-6)

    def log_likelihood(self, data):
        num_samples, num_features = data.shape
        log_probs = np.zeros((num_samples, self.num_clusters))

        for cluster in range(self.num_clusters):
            difference = data - self.cluster_means[cluster]
            var = self.cluster_var[cluster]

            log_probs[:, cluster] = (
                -0.5 * np.sum(difference ** 2 / var, axis=1)
                - 0.5 * np.sum(np.log(2 * np.pi * var))
                + np.log(self.mixing_coefficients[cluster])
            )

        log_max = log_probs.max(axis=1)
        log_likelihood = log_max + np.log(
            np.exp(log_probs - log_max[:, None]).sum(axis=1)
        )

        return log_likelihood.sum()


# --------------------------------------------------
# FIX 4: KMEANS++ INITIALIZATION
# Replaces random point selection. Spreads initial
# centers so no two start on top of each other.
# --------------------------------------------------
def kmeans_plus_plus_init(X, num_clusters, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    n_samples = X.shape[0]
    first_idx = rng.integers(0, n_samples)
    centers = [X[first_idx]]

    for _ in range(1, num_clusters):
        # Squared distance from each point to its nearest center
        dists = np.array([
            min(np.sum((x - c) ** 2) for c in centers)
            for x in X
        ])
        # Sample proportional to distance squared
        probs = dists / dists.sum()
        next_idx = rng.choice(n_samples, p=probs)
        centers.append(X[next_idx])

    return np.array(centers)


def run_em(X, num_clusters, max_iter=100, tol=1e-6, seed=None):
    rng = np.random.default_rng(seed)

    em = ExpectationMaximization(num_clusters)

    # FIX 4: kmeans++ initialization
    em.mixing_coefficients = np.ones(num_clusters) / num_clusters
    em.cluster_means = kmeans_plus_plus_init(X, num_clusters, rng)

    # FIX 1: initialize diagonal variance to per-feature variance of data
    global_var = X.var(axis=0)                      # (D,)
    em.cluster_var = np.tile(global_var, (num_clusters, 1))  # (K, D)

    log_likelihood_history = []
    cluster_mean_history = []
    responsibility_history = []
    sample_index = 0

    previous_log_likelihood = -np.inf

    for iteration in range(max_iter):

        responsibilities = em.expectation(X)
        responsibility_history.append(responsibilities[sample_index, 0])

        em.maximization(X, responsibilities)
        cluster_mean_history.append(em.cluster_means.copy())

        current_log_likelihood = em.log_likelihood(X)
        log_likelihood_history.append(current_log_likelihood)

        print(
            f"Iteration {iteration:02d} "
            f"LogLikelihood = {current_log_likelihood:.4f}"
        )

        # --------------------------------------------------
        # FIX 2: RELATIVE STOPPING CRITERION
        # Was: abs(delta) < 1e-6  → never triggers at scale 1e6
        # Now: abs(delta) / abs(previous) < 1e-6
        # --------------------------------------------------
        if iteration > 0:
            relative_change = (
                abs(current_log_likelihood - previous_log_likelihood)
                / abs(previous_log_likelihood)
            )
            if relative_change < tol:
                print(f"\nConverged at iteration {iteration} "
                      f"(relative change {relative_change:.2e})")
                break

        previous_log_likelihood = current_log_likelihood

    # --------------------------------------------------
    # FIX 3: RECOMPUTE RESPONSIBILITIES AFTER FINAL M-STEP
    # Was: using stale responsibilities from last loop iteration
    # Now: one clean E-step on the final parameters
    # --------------------------------------------------
    responsibilities = em.expectation(X)
    clusters = responsibilities.argmax(axis=1)

    return em, clusters, responsibilities, log_likelihood_history, cluster_mean_history, responsibility_history