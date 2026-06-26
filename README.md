# Expectation-Maximization (EM) for Gene Expression Clustering and Gene Regulatory Network Inference

A Python implementation of the **Expectation-Maximization (EM)** algorithm from scratch using a **Gaussian Mixture Model (GMM)** for analyzing gene expression data. This project demonstrates two levels of implementation:

1. **Gene Expression Clustering** using EM on the DREAM5 *E. coli* gene expression dataset.
2. **Gene Regulatory Network (GRN) Inference**, where clustered genes are further analyzed to infer regulatory relationships and predict network edges.

The project was developed from scratch for educational and research purposes to gain a deeper understanding of probabilistic clustering, latent variable estimation, and their application to biological network reconstruction.

---

## Features

### Level 1 – Gene Expression Clustering

* Expectation-Maximization (EM) implemented from scratch
* Gaussian Mixture Model (GMM)
* Expectation (E) Step
* Maximization (M) Step
* Gaussian likelihood computation
* Log-likelihood tracking
* Convergence monitoring
* PCA visualization of clustered gene expression profiles
* Cluster responsibility visualization
* Cluster mean evolution across iterations

### Level 2 – Gene Regulatory Network Inference

* Regulatory edge prediction using clustered gene expression profiles
* Gene–gene similarity analysis within clusters
* Construction of an inferred gene regulatory network
* Comparison of predicted interactions with the DREAM5 gold standard network
* Network visualization and edge analysis

---

## Project Structure

```text
expectation-maximization/
│
├── data/
│   ├── expression_data/
│   ├── gold_standard/
│   └── ...
│
├── images/
│
├── main.py
├── expectation_maximization.py
├── network_inference.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Goat8/expectationMaximisation.git

cd expectationMaximisation
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Level 1 – Gene Expression Clustering

Run the EM clustering algorithm:

```bash
python main.py
```

### Level 2 – Gene Regulatory Network Inference

After clustering, run the network inference module:

```bash
python network_inference.py
```

---

## Dataset

This project uses the **DREAM5 *Escherichia coli* Gene Expression Dataset**, a widely used benchmark for evaluating gene regulatory network inference algorithms.

The dataset contains expression profiles for thousands of genes measured across multiple experimental conditions.

For network evaluation, the inferred regulatory edges are compared against the provided **gold standard regulatory network**.

---

## Expectation-Maximization Algorithm

The EM algorithm iteratively estimates hidden cluster assignments and Gaussian parameters until convergence.

### Expectation (E) Step

Computes the probability (responsibility) that each gene belongs to each Gaussian component using the current parameter estimates.

Outputs:

* Responsibility matrix
* Expected cluster assignments

### Maximization (M) Step

Updates the Gaussian parameters using the computed responsibilities.

Updates:

* Mixing coefficients
* Cluster means
* Covariance (or standard deviation)
* Log likelihood

### Convergence

The algorithm stops when the improvement in log likelihood falls below a predefined threshold.

---

# Project Workflow

```text
Gene Expression Data
          │
          ▼
Expectation-Maximization
(Gaussian Mixture Model)
          │
          ▼
Gene Clusters
          │
          ▼
Gene–Gene Similarity Analysis
          │
          ▼
Predicted Regulatory Edges
          │
          ▼
Gene Regulatory Network
          │
          ▼
Comparison with DREAM5 Gold Standard
```

---

## Visualizations

The project includes visualizations for both stages of the pipeline.

### Clustering

* PCA projection of clustered genes
* Cluster responsibility heatmaps
* Log-likelihood convergence
* Cluster mean evolution

### Network Inference

* Inferred gene regulatory network
* Predicted regulatory edges
* Comparison with the DREAM5 reference network

Example:

<img width="4170" height="2959" alt="regulatory_network" src="https://github.com/user-attachments/assets/8a61e96f-176c-46f5-8221-c5f4ebf8618a" />

---

## Example Output

```text
Iteration 00  LogLikelihood = -3276.56
Iteration 01  LogLikelihood = -2145.88
Iteration 02  LogLikelihood = -1724.91
...

Converged at iteration 9

First 10 cluster assignments:
[0 0 0 0 0 0 0 0 0 0]

Number of inferred regulatory edges: 8,436
Gold standard matches: 1,127
```

---

## Future Improvements

* Full covariance Gaussian mixtures
* Automatic model selection using BIC/AIC
* Parallelized EM implementation
* Sparse graphical models for network inference
* Integration of prior biological knowledge
* Evaluation using Precision, Recall, F1-score, and AUROC

---

## License

This project is intended for educational and research purposes.
