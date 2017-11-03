"""Implements SR1 update for Hessian estimation."""
import numpy as np

def sr1_estimate(initial_hessian, mcmc, param_diff, grad_diff):
    """ Implements SR1 update for Hessian estimation.

        The limited memory SR1 algorithm is applied to estimate the Hessian
        (actually the inverse negative Hessian of the log-target) using
        gradient information used from the last memory_length number of time
        steps.

        The SR1 update is skipped if the denumerator in the update rule is
        too small. This is standard in the optimisation literature and the limit
        is determined by mcmc.settings['qn_sr1_skip_limit'].

        Args:
            initial_hessian: an estimate of the initial Hessian.
            mcmc : a Metropolis-Hastings object.
            param_diff: a list of differences in the parameters for the last
                        few iterations in the memory length.
            grad_diff:  a list of differences in the gradients for the last
                        few iterations in the memory length.

        Returns:
            First argument: estimate of the negative inverse Hessian of the
                            logarithm of the target.
            Second argument: the number of samples used to obtain the estimate.

    """
    no_samples = 0
    estimate = initial_hessian
    safe_parameterisation = mcmc.settings['qn_sr1_safe_parameterisation']

    for i in range(param_diff.shape[0]):
        diff_term = param_diff[i] - np.dot(estimate, grad_diff[i])
        term1 = np.abs(np.dot(grad_diff[i], diff_term))
        term2 = np.linalg.norm(grad_diff[i], 2)
        term2 *= np.linalg.norm(diff_term, 2)
        term2 *= mcmc.settings['qn_sr1_skip_limit']

        if term1 > term2:
            if np.dot(diff_term, grad_diff[i]) != 0.0:
                rank1_update = np.outer(diff_term, diff_term)
                rank1_update /= np.dot(diff_term, grad_diff[i])
                estimate += rank1_update
                no_samples += 1
        else:
            print("Skipping")

    if safe_parameterisation:
        estimate = np.matmul(-estimate, -estimate)
    else:
        estimate = -estimate

    return estimate, no_samples