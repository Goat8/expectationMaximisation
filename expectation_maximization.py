import numpy as np

class ExpectationMaximization:

    def __init__(
        self,
        num_clusters
    ):
        self.num_clusters = num_clusters

        self.mixing_coefficients = None
        self.cluster_means = None
        self.cluster_std = None

    def expectation(
        self,
        data
    ):
        num_samples, num_features = data.shape

        responsibilities = np.zeros(
            (num_samples, self.num_clusters)
        )

        for cluster in range(self.num_clusters):

            difference = (
                data - self.cluster_means[cluster]
            )

            likelihood = np.exp(
                -0.5 * np.sum(
                    (
                        difference
                        / self.cluster_std[cluster]
                    ) ** 2,
                    axis=1
                )
            )

            likelihood /= (
                (
                    2
                    * np.pi
                    * self.cluster_std[cluster] ** 2
                ) ** (num_features / 2)
            )

            responsibilities[:, cluster] = (
                self.mixing_coefficients[cluster]
                * likelihood
            )

        responsibilities /= responsibilities.sum(
            axis=1,
            keepdims=True
        )

        return responsibilities

 
    def maximization(
        self,
        data,
        responsibilities
    ):
        num_samples, num_features = data.shape

        self.mixing_coefficients = (
            responsibilities.sum(axis=0)
            / num_samples
        )

        self.cluster_means = (
            responsibilities.T @ data
        ) / responsibilities.sum(axis=0)[:, None]

        self.cluster_std = np.zeros(
            self.num_clusters
        )

        for cluster in range(self.num_clusters):

            difference = (
                data - self.cluster_means[cluster]
            )

            self.cluster_std[cluster] = np.sqrt(
                (
                    responsibilities[:, cluster]
                    @ (
                        difference ** 2
                    ).sum(axis=1)
                )
                /
                (
                    responsibilities[:, cluster].sum()
                    * num_features
                )
            )

    def log_likelihood(
        self,
        data
    ):
        num_samples, num_features = data.shape

        sample_probabilities = np.zeros(
            num_samples
        )

        for cluster in range(self.num_clusters):

            difference = (
                data - self.cluster_means[cluster]
            )

            likelihood = np.exp(
                -0.5 * np.sum(
                    (
                        difference
                        / self.cluster_std[cluster]
                    ) ** 2,
                    axis=1
                )
            )

            likelihood /= (
                (
                    2
                    * np.pi
                    * self.cluster_std[cluster] ** 2
                ) ** (num_features / 2)
            )

            sample_probabilities += (
                self.mixing_coefficients[cluster]
                * likelihood
            )

        return np.sum(
            np.log(sample_probabilities)
        )