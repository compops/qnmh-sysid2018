###############################################################################
#    Constructing Metropolis-Hastings proposals using damped BFGS updates
#    Copyright (C) 2018  Johan Dahlin < uni (at) johandahlin [dot] com >
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
###############################################################################

"""Helpers for computing gradients for the use in the proposal distribution
of Metropolis-Hastings algorithms."""

import numpy as np


def get_gradient(mcmc, state_estimator):
    """ Computes the gradient of the log-posterior of the parameters.

        Args:
            mcmc: an mcmc object.
            state_estimator: a state estimator object (Kalman/particle smoother)

        Returns:
            An array with the gradients of the log-posterior with respect
            to the model parameters to be estimated.

    """
    if mcmc.use_grad_info:
        gradient = state_estimator.results['gradient_internal']
    else:
        gradient = np.zeros(mcmc.model.no_params_to_estimate)

    if mcmc.settings['verbose']:
        print("Current gradient: " + str(["%.3f" % v for v in gradient]))
    return gradient


def get_nat_gradient(mcmc, gradient, inverse_hessian):
    """ Computes the natural gradient of the log-posterior of the parameters.

        The natural gradient is the product of the negative inverse Hessian
        and the gradient of the log-posterior.

        Args:
            mcmc: an mcmc object.
            state_estimator: a state estimator object (Kalman/particle smoother)
            inverse_hessian: current negative inverse Hessian of the
                             log-posterior.

        Returns:
            An array with the natural gradients of the log-posterior with
            respect to the model parameters to be estimated.

    """
    if mcmc.current_iter in mcmc.iter_hessians_corrected:
        hessian_corrected = True
    else:
        hessian_corrected = False

    if mcmc.use_grad_info:
        step_size = 0.5 * mcmc.settings['step_size']**2
        natural_gradient = np.dot(inverse_hessian, gradient)
        natural_gradient = np.array(step_size * natural_gradient).reshape(-1)
    else:
        natural_gradient = np.zeros(mcmc.model.no_params_to_estimate)

    if mcmc.settings['verbose']:
        print("Current natural gradient: " + str(["%.3f" % v for v in natural_gradient]))

    return np.real(natural_gradient)
