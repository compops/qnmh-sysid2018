"""Variance reduction methods for MCMC algorithms."""
import numpy as np

def zero_variance_linear(mcmc):
    """ Implements zero-variance post processing with linear correction. """
    no_params_to_estimate = mcmc.settings['no_params_to_estimate']
    no_iters = mcmc.settings['no_iters']
    no_burnin_iters = mcmc.settings['no_burnin_iters']

    a_hat = np.zeros((no_params_to_estimate, no_params_to_estimate))
    idx = range(no_burnin_iters, no_iters)
    gradients = -0.5 * mcmc.gradient[idx, :]
    for i in range(no_params_to_estimate):
        params = mcmc.free_params[idx, i]
        cov_params_gradients = (gradients.transpose(), params.transpose())
        cov_params_gradients = np.vstack(cov_params_gradients)
        cov_params_gradients = np.cov(cov_params_gradients)

        idx = range(no_params_to_estimate)
        cov_params_gradients_sub = cov_params_gradients[idx, idx]
        cov_params_gradients_sub = np.linalg.inv(cov_params_gradients_sub)
        cov_vector = cov_params_gradients[idx, no_params_to_estimate]
        a_hat[:, i] = - np.dot(cov_params_gradients_sub, cov_vector)

    idx = range(no_burnin_iters, no_iters)
    corrected_params = mcmc.free_params[idx, :]
    corrected_params += np.dot(gradients, a_hat)
    return corrected_params
