"""Implements BFGS update for Hessian estimation."""
import numpy as np

def bfgs_estimate(initial_hessian, mcmc, param_diff, grad_diff):
    """ Implements BFGS update for Hessian estimation.

        The limited memory BFGS algorithm is applied to estimate the Hessian
        (actually the inverse negative Hessian of the log-target) using
        gradient information used from the last memory_length number of time
        steps.

        The curvature condition in the BFGS algorithm is important as it
        makes sure that the estimate is positive semi-definite. It can be
        controlled by setting the field mcmc.settings['qn_bfgs_curvature_cond']:

            'enforce': the standard condition is enforced and all differences
                       in parameters and gradients violating this condition
                       are removed.
            'ignore':  ignores the condition and relies on a correction in
                       a later step to obtain a positive semidefinite estimate.
            'damped':  makes use of damped BFGS to adjust the differences in
                       parameters and gradients to fulfill the curvature
                       condition.

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
    estimate = initial_hessian
    curv_cond = mcmc.settings['qn_bfgs_curvature_cond']
    identity_matrix = np.diag(np.ones(mcmc.model.no_params_to_estimate))
    no_samples = 0
    violate_curv_cond = 0

    for i in range(param_diff.shape[0]):
        do_update = False

        if curv_cond is 'enforce':
            param_diff[i] = -param_diff[i]
            if np.dot(param_diff[i], grad_diff[i]) > 0.0:
                do_update = True
                new_grad_diff = grad_diff[i]
            else:
                violate_curv_cond += 1

        elif curv_cond is 'damped':
            param_diff[i] = -param_diff[i]
            inverse_hessian = np.linalg.inv(estimate)
            term1 = np.dot(param_diff[i], grad_diff[i])
            term2 = np.dot(param_diff[i], inverse_hessian)
            term2 = np.dot(term2, param_diff[i])
            if term1 > 0.2 * term2:
                theta = 1.0
            else:
                theta = 0.8 * term2 / (term2 - term1)

            grad_guess = np.dot(inverse_hessian, param_diff[i])
            new_grad_diff = theta * grad_diff[i] + (1.0 - theta) * grad_guess
            do_update = True

        elif curv_cond is 'ignore':
            do_update = True
            new_grad_diff = grad_diff[i]
        else:
            raise NameError("Unknown flag curv_cond given to function")

        if do_update:
            no_samples += 1
            rho = 1.0 / np.dot(new_grad_diff, param_diff[i])
            term1 = np.outer(param_diff[i], new_grad_diff)
            term1 = identity_matrix - rho * term1
            term2 = np.outer(new_grad_diff, param_diff[i])
            term2 = identity_matrix - rho * term2
            term3 = rho * np.outer(param_diff[i], param_diff[i])

            tmp_term1 = np.matmul(term1, estimate)
            estimate = np.matmul(tmp_term1, term2) + term3

        if curv_cond is 'enforce' or curv_cond is 'ignore':
            estimate = -estimate
    return estimate, no_samples