# Extreme Learning Machines: Numerical Solvers

This repository contains a MATLAB implementation of an Extreme Learning Machine (ELM) for regression tasks. The primary focus of this project is the efficient and stable numerical resolution of the ELM training phase, which translates to a Tikhonov-regularized linear least-squares problem.

## Implemented Solvers

To find the optimal output weights of the neural network, three different algorithmic approaches have been implemented from scratch:

1. **Thin QR Factorization (`qr`)**: A direct solver that uses Householder reflectors on an augmented system matrix. It avoids forming the Gram matrix, thus preserving the condition number and ensuring high numerical stability.
2. **Conjugate Gradient (`cg`)**: An iterative Krylov subspace method applied to the regularized Normal Equations. It leverages fast matrix-matrix multiplications, making it highly time-efficient for large-scale, well-conditioned datasets.
3. **LSQR (`lsqr`)**: An iterative solver based on Golub-Kahan bidiagonalization (Paige & Saunders, 1982). It provides the speed of iterative methods like CG but operates directly on the augmented matrix, preventing the catastrophic loss of precision associated with ill-conditioned Normal Equations.

## Repository Structure

- `src/algorithms/`: Core numerical linear algebra implementations (`thin_qr.m`, `conjugate_gradient.m`, `lsqr.m`, etc.).
- `src/utils/`: Machine learning utility functions for the ELM wrapper (`elm_train.m`, `elm_predict.m`, `apply_Qt.m`).
- `src/`: Testing and benchmarking scripts (`test_comparison.m`, `n_neurons_vs_n_samples.m`, etc.) used to evaluate time complexity, iterations, and test errors.
- `docs/`: Generated plots and figures analyzing the algorithmic scalability and generalization capabilities.

## Usage

To train and evaluate an ELM model, ensure that the `src` folder and its subdirectories are added to your MATLAB path.

```matlab
% Define hyperparameters
hidden_dim = 150;
lambda = 1e-3;
activation = @(z) tanh(z);

% Train the model using one of the solvers: 'qr', 'cg', or 'lsqr'
model = elm_train(X_train, Y_train, hidden_dim, lambda, activation, 'lsqr');

% Make predictions on new data
Y_pred = elm_predict(model, X_test);
```

To reproduce the experimental results and plots discussed in the project report, simply run the benchmarking scripts located in the root of the `src/` directory (e.g., `test_comparison.m`).

# Requirements
- **MATLAB**: The code relies on base MATLAB functionality. No additional toolboxes (like the Machine Learning or Optimization toolboxes) are strictly required to run the core algorithms.
