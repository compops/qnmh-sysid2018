"""Helpers for checking covariance matrices."""

import numpy as np
from scipy.linalg import eigh
from scipy.stats._multivariate import _eigvalsh_to_eps

def is_psd(cov_matrix):
    """Check if a matrix is positive definite (PD) by checking for
    negative eigenvalues. Returns True if PD and False otherwise."""
    return np.all(np.linalg.eigvals(cov_matrix) > 0)

def is_valid_covariance_matrix(cov_matrix):
    """Check if a matrix is positive definite (PD) by checking for
    negative eigenvalues and also if the matrix is ill conditioned (close to
    singular). Returns True if PD and well conditioned and False otherwise."""
    eig_values = eigh(cov_matrix, lower=True, check_finite=True)[0]

    eps = _eigvalsh_to_eps(eig_values, None, None)
    if np.min(eig_values) < -eps:
        return False
    large_eig_values = eig_values[eig_values > eps]
    if len(large_eig_values) < len(eig_values):
        return False
    return True
