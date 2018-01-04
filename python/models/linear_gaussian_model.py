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

"""System model class for a linear Gaussian state-space model."""
import numpy as np
from scipy.stats import norm

from models.base_model import BaseModel
from helpers.distributions import normal
from helpers.distributions import gamma


class LinearGaussianModel(BaseModel):
    """ System model class for a linear Gaussian state-space model.

        Encodes the model with the parameterisation:

        x[t+1] = mu + phi * (x[t] - mu) + sigma[v] * v[t]
        y[t]   = x[t]                   + sigma[e] * e[t]

        where v[t], e[t] are independent and standard Gaussian i.e., N(0, 1).
        The parameters of the model are (mu, phi, sigma[v], sigma[e]).

        Furthermore, the inference model is reparameterised to enable all
        parameters to be unrestricted (can assume any real value) in the
        inference step. Hence, the reparameterised model is given by

        mu = mu
        phi = tanh(eta)
        sigma[v] = exp(psi)
        sigma[e] = exp(tau)

        where (eta, psi, tau) now as assume any real value. This results in that
        this transformation needs to be taken into account when computing
        gradients as well as in the Jacobian.

    """

    def __init__(self):
        self.name = "Linear Gaussian state-space model with four parameters."
        self.file_prefix = "linear_gaussian_model"

        self.params = {'mu': 0.2, 'phi': 0.75, 'sigma_v': 1.0, 'sigma_e': 0.1}
        self.free_params = {'mu': 0.2, 'phi': 0.64, 'sigma_v': 0.0,
                            'sigma_e': -2.3}
        self.no_params = len(self.params)
        self.params_prior = {'mu': (normal, 0.0, 1.0),
                             'phi': (normal, 0.5, 1.0),
                             'sigma_v': (gamma, 2.0, 2.0),
                             'sigma_e': (gamma, 2.0, 2.0)
                             }
        self.intial_state = []
        self.no_obs = []
        self.states = []
        self.inputs = []
        self.obs = []
        self.params_to_estimate_idx = []
        self.no_params_to_estimate = 0
        self.params_to_estimate = []
        self.true_params = []

    def generate_initial_state(self, no_samples):
        """ Generates no_samples from the initial state distribution.

            Args:
                no_samples: number of samples to generate (integer).

            Returns:
                An array with no_samples from the initial state distribution.

        """
        mean = self.params['mu']
        noise_stdev = self.params['sigma_v']
        noise_stdev /= np.sqrt(1.0 - self.params['phi']**2)
        return mean + noise_stdev * np.random.normal(size=(1, no_samples))

    def generate_state(self, cur_state, time_step):
        """ Generates a new state by the state dynamics.

            Args:
                cur_state: the current state (array).
                time_step: the current time step (integer).

            Returns:
                An array ofsamples from the next time step.

        """
        mean = self.params['mu']
        mean += self.params['phi'] * (cur_state - self.params['mu'])
        noise_stdev = self.params['sigma_v']
        noise = noise_stdev * np.random.randn(1, len(cur_state))
        return mean + noise

    def evaluate_state(self, next_state, cur_state, time_step):
        """ Computes the probability of a state transition.

            Args:
                next_state: the next state (array)
                cur_state: the current state (array).
                time_step: the current time step (integer).

            Returns:
                An array of transition log-probabilities.

        """
        mean = self.params['mu']
        mean += self.params['phi'] * (cur_state - self.params['mu'])
        stdev = self.params['sigma_v']
        return norm.logpdf(next_state, mean, stdev)

    def generate_obs(self, cur_state, time_step):
        """ Generates a new observation by the observation dynamics.

            Args:
                cur_state: the current state (array).
                time_step: the current time step (integer).

            Returns:
                An array of observations.

        """
        mean = cur_state
        noise_stdev = self.params['sigma_e']
        noise = noise_stdev * np.random.randn(1, len(cur_state))
        return mean + noise

    def evaluate_obs(self, cur_state, time_step):
        """ Computes the probability of obtaining an observation.

            Args:
                cur_state: the current state (array).
                time_step: the current time step (integer).

            Returns:
                An array of observation log-probabilities.

        """
        current_obs = self.obs[time_step]
        return norm.logpdf(current_obs, cur_state, self.params['sigma_e'])

    def check_parameters(self):
        """" Checks if parameters satisfies hard constraints on the parameters.

                Returns:
                    Boolean to indicate if the current parameters results in
                    a stable system and obey the constraints on their values.

        """
        if np.abs(self.params['phi']) > 1.0:
            parameters_are_okey = False
        elif self.params['sigma_v'] < 0.0 or self.params['sigma_e'] < 0.0:
            parameters_are_okey = False
        else:
            parameters_are_okey = True
        return parameters_are_okey

    def log_prior_gradient(self):
        """ Returns the logarithm of the prior distribution.

            Returns:
                First value: a dict with an entry for each parameter.
                Second value: the sum of the log-prior for all variables.

        """
        gradients = super(LinearGaussianModel, self).log_prior_gradient()

        gradients['phi'] *= (1.0 - self.params['phi']**2)
        gradients['sigma_v'] *= self.params['sigma_v']
        gradients['sigma_e'] *= self.params['sigma_e']
        return gradients

    def log_prior_hessian(self):
        """ The Hessian of the logarithm of the prior.

            Returns:
                A dict with an entry for each parameter.

        """
        gradients = super(LinearGaussianModel, self).log_prior_gradient()
        hessians = super(LinearGaussianModel, self).log_prior_hessian()

        gradients['phi'] *= (1.0 - 2.0 * self.params['phi']**2)
        hessians['phi'] *= (1.0 - self.params['phi']**2)
        hessians['phi'] += gradients['phi']
        gradients['sigma_v'] *= self.params['sigma_v']
        hessians['sigma_v'] *= self.params['sigma_v']
        hessians['sigma_v'] += gradients['sigma_v']
        gradients['sigma_e'] *= self.params['sigma_e']
        hessians['sigma_e'] *= self.params['sigma_e']
        hessians['sigma_e'] += gradients['sigma_e']
        return hessians

    def log_joint_gradient(self, next_state, cur_state, time_index):
        """ The gradient of the joint distribution of observations and states.

            Computes the gradient of log p(x, y) for use in Fisher's identity
            to compute the gradient of the log-likelihood.

            Args:
                next_state: the next state. (array)
                cur_state: the current state. (array)
                time_index: the current time index. (integer)

            Returns:
                A dict with an entry for each parameter.

        """
        current_obs = self.obs[time_index]

        state_quad_term = next_state - self.params['mu']
        state_quad_term -= self.params['phi'] * (cur_state - self.params['mu'])
        obs_quad_term = current_obs - cur_state

        q_matrix = self.params['sigma_v']**(-2)
        r_matrix = self.params['sigma_e']**(-2)

        gradient_mu = q_matrix * state_quad_term * (1.0 - self.params['phi'])
        gradient_phi = q_matrix * state_quad_term
        gradient_phi *= (cur_state - self.params['mu'])
        gradient_phi *= (1.0 - self.params['phi']**2)
        gradient_sigmav = q_matrix * state_quad_term**2 - 1.0
        gradient_sigmae = r_matrix * obs_quad_term**2 - 1.0

        gradient = {}
        gradient.update({'mu': gradient_mu})
        gradient.update({'phi': gradient_phi})
        gradient.update({'sigma_v': gradient_sigmav})
        gradient.update({'sigma_e': gradient_sigmae})
        return gradient

    def transform_params_to_free(self):
        """ Computes and store the values of the reparameterised parameters.

            These transformations are dictated directly from the model. See
            the docstring for the model class for more information. The
            values of the reparameterised parameters are computed by applying
            the transformation to the current standard parameters stored in
            the model object.

        """
        self.free_params['mu'] = self.params['mu']
        self.free_params['phi'] = np.arctanh(self.params['phi'])
        self.free_params['sigma_v'] = np.log(self.params['sigma_v'])
        self.free_params['sigma_e'] = np.log(self.params['sigma_e'])

    def transform_params_from_free(self):
        """ Computes and store the values of the standard parameters.

            These transformations are dictated directly from the model. See
            the docstring for the model class for more information. The
            values of the standard parameters are computed by applying
            the transformation to the current reparameterised parameters stored
            in the model object.

        """
        self.params['mu'] = self.free_params['mu']
        self.params['phi'] = np.tanh(self.free_params['phi'])
        self.params['sigma_v'] = np.exp(self.free_params['sigma_v'])
        self.params['sigma_e'] = np.exp(self.free_params['sigma_e'])

    def log_jacobian(self):
        """ Computes the sum of the log-Jacobian.

            These Jacobians are dictated by the transformations for the model.
            See the docstring for the model class for more information.

            Returns:
                the sum of the logarithm of the Jacobian of the parameter
                transformation for the parameters under inference as listed
                in params_to_estimate.

        """
        try:
            jacobian = {}
            jacobian.update({'mu': 0.0})
            jacobian.update({'phi': np.log(1.0 - self.params['phi']**2)})
            jacobian.update({'sigma_v': np.log(self.params['sigma_v'])})
            jacobian.update({'sigma_e': np.log(self.params['sigma_e'])})
        except RuntimeWarning:
            return -np.inf

        return self._compile_log_jacobian(jacobian)

