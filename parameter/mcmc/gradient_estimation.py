"""Helpers for computing gradients for the use in the proposal distribution
of Metropolis-Hastings algorithms."""

import numpy as np

def get_gradient(mcmc, state_estimator):
    """Computes the gradient of the log-posterior of the parameters."""
    if mcmc.use_gradient_information:
        gradient = state_estimator.gradient_internal
    else:
        gradient = np.zeros(mcmc.model.no_params_to_estimate)

    if mcmc.settings['verbose']:
        print("Current gradient: " + str(gradient) + ".")
    return gradient

def get_nat_gradient(mcmc, gradient, inverse_hessian):
    """Computes the natural gradient (the gradient scaled by the negative
    inverse Hessian) of the log-posterior of the parameters."""
    gradient_requested = False
    hessian_corrected = False

    if mcmc.settings['qn_memory_length']:
        if mcmc.current_iter > mcmc.settings['qn_memory_length']:
            gradient_requested = True
    else:
        gradient_requested = True

    if mcmc.current_iter in mcmc.iter_hessians_corrected:
        hessian_corrected = True

    if mcmc.settings['qn_sr1_safe_parameterisation'] and hessian_corrected:
        inverse_hessian = np.linalg.cholesky(inverse_hessian)

    if mcmc.use_gradient_information and gradient_requested:
        step_size = 0.5 * mcmc.settings['step_size']**2
        natural_gradient = np.dot(inverse_hessian, gradient)
        natural_gradient = np.array(step_size * natural_gradient).reshape(-1)
    else:
        natural_gradient = np.zeros(mcmc.model.no_params_to_estimate)

    if mcmc.settings['verbose']:
        print("Current natural gradient: " + str(natural_gradient) + ".")
    return natural_gradient
