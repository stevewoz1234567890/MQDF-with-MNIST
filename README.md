# MQDF with MNIST

Experimental notebook that trains a **Modified Quadratic Discriminant Function (MQDF)** classifier on the [MNIST](http://yann.lecun.com/exdb/mnist/) digit dataset. Per-class Gaussian models use regularized covariance matrices \(\Sigma_i + h^2 I\); eigen-decomposition of each matrix yields the MQDF score used for classification.

## Contents

| File | Description |
|------|-------------|
| `MQDF.ipynb` | End-to-end pipeline: load MNIST, estimate means/covariances, build MQDF, evaluate on the test set, sweep the regularization hyperparameter `h`. |

## Requirements

- Python 3
- `numpy`, `matplotlib`

The notebook also uses `google.colab` and Google Drive in the first cells so paths match a Colab + Drive layout. For local use, replace those cells with paths to your MNIST files (see below).

## MNIST data

You need the four standard IDX files (often distributed as `.gz`):

- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`

Point the `gzip.open(...)` calls in the notebook at the directory where these files live.

## What the notebook does

1. **Load** training and test images (28×28 flattened to 784) and labels.
2. **Estimate** per-class mean \(\mu_i\) and covariance \(\Sigma_i\) from the training set.
3. **Regularize** covariances as \(\Sigma_i + h^2 I\) for a scalar \(h > 0\), then **eigen-decompose** each matrix.
4. **Classify** each test sample by the class that maximizes the MQDF discriminant (implemented via the eigenbasis).
5. **Tune** `h` over a small grid, record accuracy, and plot accuracy vs. `h` (saved as `result.png` in the notebook’s working directory).

## Running

**Google Colab:** Run cells in order. Mount Drive when prompted and ensure the MNIST archives are in the folder referenced by the notebook (or update the paths).

**Locally:** Install dependencies, open `MQDF.ipynb` in Jupyter or VS Code, remove or skip the Colab/Drive cells, set MNIST paths, then run the remaining cells.
