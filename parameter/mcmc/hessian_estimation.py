"""Helpers for computing Hessians for the use in the proposal distribution
of Metropolis-Hastings algorithms."""

import numpy as np
from helpers.cov_matrix import is_psd
from parameter.mcmc.quasi_newton.main import quasi_newton

def get_hessian(mcmc, state_estimator, prop_gradient=None):
    """Get the Hessian of the log-posterior of the parameters as a fixed
    matrix or estimate it using Kalman smoothing or Quasi-Newton methods.

    Args:
        mcmc: Metropolis-Hastings sampler object
        state_estimator: object for estimating the Hessian using Kalman or
                         particle smoothing methods.
        prop_gradient: the current proposed gradient (to scale initial Hessian)

    Returns:
        An estimate of the negative inverse Hessian of the log-posterior.
    """
    step_size = mcmc.settings['step_size']**2

    # Default choice for Hessian from user
    inverse_hessian = step_size * mcmc.settings['base_hessian']

    # Estimate Hessian using Kalman smoothing or Quasi-Newton methods
    if mcmc.use_hessian_information:
        if mcmc.settings['hessian_estimate'] is 'kalman':
            inverse_hessian = np.linalg.inv(state_estimator.hessian_internal)
            inverse_hessian *= step_size
            return _correct_hessian(inverse_hessian, mcmc)
        if mcmc.settings['hessian_estimate'] is 'quasi_newton':
            if mcmc.current_iter > mcmc.settings['qn_memory_length']:
                inverse_hessian, no_samples = quasi_newton(mcmc, prop_gradient)
                inverse_hessian *= step_size
                mcmc.no_samples_hess_est[mcmc.current_iter] = no_samples
                return _correct_hessian(inverse_hessian, mcmc)

    if mcmc.settings['verbose']:
        print("Current inverse_hessian: " + str(inverse_hessian) + ".")
    return inverse_hessian

def _correct_hessian(estimate, mcmc):
    """Corrects a Hessian estimate that is not postive definite.

    Args:
        estimate: an estimate of the negative inverse Hessian of the
                  log-posterior.
        mcmc: Metropolis-Hastings sampler object

    Returns:
        A corrected estimate (positive definite) of the negative inverse Hessian
        of the log-posterior.
    """
    strategy = mcmc.settings['hessian_correction']

    # No correction
    if not strategy:
        return estimate

    if estimate is None or not is_psd(estimate):
        mcmc.no_hessians_corrected += 1
        mcmc.iter_hessians_corrected.append(mcmc.current_iter)

        # if is_psd(-estimate):
        #     print("Iteration: " + str(mcmc.current_iter) +
        #           ", switched to negative Hessian estimate...")
        #     return -estimate

        if strategy is 'replace' or estimate is None:
            if mcmc.current_iter > mcmc.settings['no_burnin_iters']:
                if mcmc.settings['hessian_correction_verbose']:
                    print("Iteration: " + str(mcmc.current_iter) +
                          ", corrected Hessian by replacing with estimate from " +
                          "latter half of burn-in.")

                if not hasattr(mcmc, 'emp_hessian'):
                    idx = range(int(0.5 * mcmc.settings['no_burnin_iters']),
                                mcmc.settings['no_burnin_iters'])
                    trace = mcmc.free_params[idx, :]
                    mcmc.emp_hessian = np.cov(trace, rowvar=False)
                    print("Iteration: " + str(mcmc.current_iter) +
                          ", computed an empirical estimate of the posterior "
                          + "covariance to replace ND Hessian estimates.")
                return mcmc.emp_hessian
            else:
                step_size = mcmc.settings['step_size']**2
                return step_size * mcmc.settings['base_hessian']

        # Add a diagonal matrix proportional to the largest negative eigenvalue
        elif strategy is 'regularise':
            min_eigval = np.min(np.linalg.eig(estimate)[0])
            if mcmc.settings['hessian_correction_verbose']:
                print("Iteration: " + str(mcmc.current_iter) +
                      ", corrected Hessian by adding diagonal matrix " +
                      " with elements: " + str(-2.0 * min_eigval))
            return estimate - 2.0 * min_eigval * np.eye(estimate.shape[0])

        # Flip the negative eigenvalues
        elif strategy is 'flip':
            if mcmc.settings['hessian_correction_verbose']:
                print("Iteration: " + str(mcmc.current_iter) +
                      ", corrected Hessian by flipping negative eigenvalues " +
                      "to positive.")
            evd = np.linalg.eig(estimate)
            ev_matrix = np.diag(np.abs(evd[0]))
            return np.dot(np.dot(evd[1], ev_matrix), evd[1])
        else:
            raise ValueError("Unknown Hessian correction strategy...")
    else:
        return estimate