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

    def expectation(self, data):
      num_samples, num_features = data.shape
      log_responsibilities = np.zeros(
          (num_samples, self.num_clusters)
      )
      
      for cluster in range(self.num_clusters):
          difference = data - self.cluster_means[cluster]
          
          # Compute log likelihood directly — avoids overflow
          log_likelihood = (
              -0.5 * np.sum(
                  (difference / self.cluster_std[cluster]) ** 2,
                  axis=1
              )
              - (num_features / 2) * np.log(
                  2 * np.pi * self.cluster_std[cluster] ** 2
              )
              + np.log(self.mixing_coefficients[cluster])
          )
          
          log_responsibilities[:, cluster] = log_likelihood
      
      # Log-sum-exp trick for numerical stability
      log_max = log_responsibilities.max(axis=1, keepdims=True)
      log_responsibilities -= log_max
      responsibilities = np.exp(log_responsibilities)
      responsibilities /= responsibilities.sum(
          axis=1, keepdims=True
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

    def log_likelihood2(
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

    def log_likelihood(self, data):
        num_samples, num_features = data.shape
        log_probs = np.zeros(
            (num_samples, self.num_clusters)
        )
        
        for cluster in range(self.num_clusters):
            difference = data - self.cluster_means[cluster]
            log_probs[:, cluster] = (
                -0.5 * np.sum(
                    (difference / self.cluster_std[cluster]) ** 2,
                    axis=1
                )
                - (num_features / 2) * np.log(
                    2 * np.pi * self.cluster_std[cluster] ** 2
                )
                + np.log(self.mixing_coefficients[cluster])
            )
        
        # Log-sum-exp per sample then sum
        log_max = log_probs.max(axis=1)
        log_likelihood = (
            log_max
            + np.log(
                np.exp(log_probs - log_max[:, None]).sum(axis=1)
            )
        )
        
        return log_likelihood.sum()