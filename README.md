# Extreme Learning Machine Implementation
## Computational Mathematics Project

### Overview
This project implements an **Extreme Learning Machine (ELM)** algorithm in MATLAB, focusing on efficient numerical linear algebra techniques. As per the project requirements, the training phase involves solving a linear least-squares problem with $L_2$ regularization using two distinct approaches:

1.  **Direct Method:** Utilizing **Thin QR Factorization** (scaling linearly with dimensions).
2.  **Iterative Method:** Utilizing the **Conjugate Gradient (CG)** method.

### Mathematical Formulation
The model is defined as a single-hidden layer neural network:
$$y = W_2 \sigma(W_1 x)$$

Where:
* $W_1$ is a fixed random weight matrix.
* $\sigma(\cdot)$ is the element-wise activation function.
* $W_2$ is the output weight matrix computed by solving:
    $$\min_{W_2} \| H W_2^T - Y \|_2^2 + \lambda \| W_2 \|_2^2$$

### Algorithms
The repository contains custom implementations (no built-in `backslash` for the core solvers) of:

* **Thin QR Factorization:** Implemented via Householder reflectors for numerical stability.
* **Conjugate Gradient:** Implemented for solving the normal equations $A^T A x = A^T b$.

### Structure
* `src/algorithms/`: Core numerical methods (`thin_qr.m`, `conjugate_gradient.m`).
* `src/utils/`: Helper functions for managing the compilation.

### Performance Analysis
The project includes a study on how the **hidden layer dimension** affects:
* Computational time (CPU time).
* Accuracy (RMSE).

*(futuro un grafico dei risultati generato da MATLAB)*

### 👥 Authors
* Davide Fantasia
* Massimo Parlanti
