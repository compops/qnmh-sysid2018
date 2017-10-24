"""Helpers for computing gradients for the use in the proposal distribution
of Metropolis-Hastings algorithms."""

import numpy as np

def get_gradient(sampler, state_estimator):
    """Computes the gradient of the log-posterior of the parameters."""
    if sampler.use_gradient_informration:
        gradient = state_estimator.gradient_internal
    else:
        gradient = np.zeros(sampler.settings['noParametersToEstimate'])

    if sampler.settings['verbose']:
        print("Current gradient: " + str(gradient) + ".")
    return gradient

def get_nat_gradient(sampler, gradient, inverse_hessian):
    """Computes the natural gradient (the gradient scaled by the negative
    inverse Hessian) of the log-posterior of the parameters."""
    flag = False
    if sampler.settings['memoryLength']:
        if sampler.current_iter > sampler.settings['memoryLength']:
            flag = True
    else:
        flag = True

    if sampler.use_gradient_informration and flag:
        step_size = 0.5 * sampler.settings['step_size']**2
        natural_gradient = np.dot(inverse_hessian, gradient)
        natural_gradient = np.array(step_size * natural_gradient).reshape(-1)
    else:
        natural_gradient = np.zeros(sampler.settings['noParametersToEstimate'])

    if sampler.settings['verbose']:
        print("Current natural gradient: " + str(natural_gradient) + ".")
    return natural_gradient
