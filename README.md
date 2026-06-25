# Expectation-Maximization (EM) Algorithm from Scratch

A Python implementation of the **Expectation-Maximization (EM)** algorithm for clustering synthetic gene expression data using a **Gaussian Mixture Model (GMM)**.

This project was built from scratch for educational purposes to better understand how the EM algorithm estimates hidden cluster assignments and model parameters without relying on machine learning libraries.

---

## Features

- Implementation of the EM algorithm from scratch
- Expectation (E) Step
- Maximization (M) Step
- Gaussian likelihood computation
- Log-likelihood tracking
- Convergence monitoring
- PCA visualization before and after clustering
- Cluster responsibility visualization
- Cluster mean evolution across iterations

---

## Project Structure

```
expectation-maximization/
│
├── main.py
├── expectation_maximization.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/expectation-maximization.git

cd expectation-maximization
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run

```bash
python main.py
```

---

## Dataset

The project generates a synthetic gene expression dataset consisting of two Gaussian clusters.

```python
X = np.vstack([
    np.random.randn(50, 10) + 2,
    np.random.randn(50, 10) - 2
])
```

- 100 samples
- 10 features (gene expression values)
- 2 hidden clusters

---

## EM Algorithm

The algorithm alternates between two steps until convergence.

### Expectation Step

Computes the probability that each sample belongs to every cluster.

- Uses the current Gaussian parameters
- Produces the responsibility matrix

### Maximization Step

Updates the Gaussian parameters using the responsibilities.

Updates:

- Mixing coefficients
- Cluster means
- Cluster standard deviations

### Convergence

The algorithm monitors the log likelihood of the data and stops when the improvement becomes negligible.

---

## Visualizations

The project generates several plots:

- Log Likelihood vs Iteration
- Responsibility Evolution
- Cluster Mean Evolution
- PCA Projection Before EM
- PCA Projection After EM

---

## Example Output

```
Iteration 00 LogLikelihood = -3276.56
Iteration 01 LogLikelihood = -2145.88
Iteration 02 LogLikelihood = -1724.91
...

Converged at iteration 9

First 10 assignments:
[0 0 0 0 0 0 0 0 0 0]
```

---

## Learning Objectives

This project demonstrates:
- Expectation-Maximization (EM)
- Soft clustering
- Maximum likelihood estimation
- Multivariate Gaussian distributions
- Probability-based clustering

---

## Future Improvements

- Support multiple covariance matrices
- Random parameter initialization
- K-Means initialization
- Predict method for unseen samples
- Model selection using AIC/BIC
- Real-world gene expression datasets
- Gene regulatory network inference

---

## License

This project is intended for educational and research purposes.
