"""Implements SR1 update for Hessian estimation."""
import numpy as np

def sr1_estimate(estimate, param_diff, grad_diff):
    """Implements SR1 update for Hessian estimation."""
    no_samples = 0

    for i in range(param_diff.shape[0]):
        diff_term = param_diff[i] - np.dot(estimate, grad_diff[i])
        if np.dot(diff_term, grad_diff[i]) != 0.0:
            rank1_update = np.outer(diff_term, diff_term)
            rank1_update /= np.dot(diff_term, grad_diff[i])
            estimate += rank1_update
            no_samples += 1

    return -estimate, no_samples