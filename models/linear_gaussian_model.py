"""System model class for a linear Gaussian state-space model."""
import numpy as np
from scipy.stats import norm

from helpers.data_generation import generate_data, import_data
from helpers.inference_model import create_inference_model, fix_true_params
from helpers.model_params import store_free_params, store_params
from helpers.model_params import get_free_params, get_params
from helpers.distributions import normal
from helpers.distributions import gamma

class SystemModel(object):
    """System model class for a linear Gaussian state-space model."""

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
        self.intial_state = 0.0
        self.no_obs = 500
        self.states = []
        self.obs = []
        self.params_to_estimate_idx = []
        self.no_params_to_estimate = 0
        self.params_to_estimate = []

    def generate_initial_state(self, no_samples):
        """Generates no_samples from the initial state distribution."""
        mean = self.params['mu']
        noise_stdev = self.params['sigma_v']
        noise_stdev /= np.sqrt(1.0 - self.params['phi']**2)
        return mean + noise_stdev * np.random.normal(size=(1, no_samples))

    def generate_state(self, current_state):
        """Generates a new state by the state dynamics given current_state."""
        mean = self.params['mu']
        mean += self.params['phi'] * (current_state - self.params['mu'])
        noise_stdev = self.params['sigma_v']
        noise = noise_stdev * np.random.randn(1, len(current_state))
        return mean + noise

    def evaluate_state(self, next_state, current_state):
        """Evaluates the probability of current_state and next_state under
           the state dynamics model."""
        mean = self.params['mu']
        mean += self.params['phi'] * (current_state - self.params['mu'])
        stdev = self.params['sigma_v']
        return norm.logpdf(next_state, mean, stdev)

    def generate_obs(self, current_state):
        """Generates a new observation by the observation dynamics
           given current_state."""
        mean = current_state
        noise_stdev = self.params['sigma_e']
        noise = noise_stdev * np.random.randn(1, len(current_state))
        return mean + noise

    def evaluate_obs(self, current_state, current_obs):
        """Evaluates the probability of current_state and current_obs
           under the observation model."""
        mean = current_state
        stdev = self.params['sigma_e']
        return norm.logpdf(current_obs, mean, stdev)

    def check_parameters(self):
        """"Checks if parameters satisifes hard constraints."""
        if np.abs(self.params['phi']) > 1.0:
            parameters_are_okey = False
        elif self.params['sigma_v'] < 0.0 or self.params['sigma_e'] < 0.0:
            parameters_are_okey = False
        else:
            parameters_are_okey = True
        return parameters_are_okey

    def log_prior(self):
        """Returns the logarithm of the prior distribution as a dictionary
        with a member for each parameter and the sum as a float."""
        prior = {}
        for param in self.params:
            dist = self.params_prior[param][0]
            hyppar1 = self.params_prior[param][1]
            hyppar2 = self.params_prior[param][2]
            prior.update({param: dist.logpdf(self.params[param], hyppar1, hyppar2)})

        prior_sum = 0.0
        for param in self.params_to_estimate:
            prior_sum += prior[param]

        return prior, prior_sum

    def log_prior_gradient(self):
        """Returns the gradient of the logarithm of the prior distributions as
        a dictionary with a member for each parameter."""
        gradients = {}
        for param in self.params:
            dist = self.params_prior[param][0]
            value = self.params[param]
            hyppar1 = self.params_prior[param][1]
            hyppar2 = self.params_prior[param][2]
            gradient = dist.logpdf_gradient(value, hyppar1, hyppar2)
            gradients.update({param: gradient})

        gradients['phi'] *= (1.0 - self.params['phi']**2)
        gradients['sigma_v'] *= self.params['sigma_v']
        gradients['sigma_e'] *= self.params['sigma_e']
        return gradients

    def log_prior_hessian(self):
        """Returns the Hessian of the logarithm of the prior distributions as
        a dictionary with a member for each parameter."""
        hessians = {}
        gradients = {}
        for param in self.params:
            dist = self.params_prior[param][0]
            value = self.params[param]
            hyppar1 = self.params_prior[param][1]
            hyppar2 = self.params_prior[param][2]
            gradient = dist.logpdf_gradient(value, hyppar1, hyppar2)
            hessian = dist.logpdf_hessian(value, hyppar1, hyppar2)
            gradients.update({param: gradient})
            hessians.update({param: hessian})

        gradients['phi'] *= (1.0 - 2.0 * self.params['phi']**2)
        hessians['phi'] *= (1.0 - self.params['phi']**2)
        hessians['phi'] += gradients['phi']
        gradients['sigma_v'] *= self.params['sigma_v']
        hessians['sigma_v'] *= self.params['sigma_v']
        hessians['sigma_v'] += gradients['sigma_v']

        return hessians

    def log_joint_gradient(self, next_state, current_state, current_obs):
        """Returns the gradient of the logarithm of the joint distributions of
        states and observations as a dictionary with a member for each
        parameter."""
        state_quad_term = current_state - self.params['mu']
        state_quad_term *= self.params['phi']
        state_quad_term += next_state - self.params['mu']
        obs_quad_term = current_obs - current_state
        q_matrix = self.params['sigma_v']**(-2)
        r_matrix = self.params['sigma_e']**(-2)

        gradient_mu = q_matrix * state_quad_term * (1.0 - self.params['phi'])
        gradient_phi = q_matrix * state_quad_term
        gradient_phi *= (current_state - self.params['mu'])
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
        """Transforms the current parameters to their free parameterisation."""
        self.free_params['mu'] = self.params['mu']
        self.free_params['phi'] = np.arctanh(self.params['phi'])
        self.free_params['sigma_v'] = np.log(self.params['sigma_v'])
        self.free_params['sigma_e'] = np.log(self.params['sigma_e'])

    def transform_params_from_free(self):
        """Get the model parameters by transforming from their free
        parameterisation."""
        self.params['mu'] = self.free_params['mu']
        self.params['phi'] = np.tanh(self.free_params['phi'])
        self.params['sigma_v'] = np.exp(self.free_params['sigma_v'])
        self.params['sigma_e'] = np.exp(self.free_params['sigma_e'])

    def log_jacobian(self):
        """Returns the sum of the log-Jacobian for all parameters to be
        estimated."""
        jacobian = {}
        jacobian.update({'mu': 0.0})
        jacobian.update({'phi': np.log(1.0 - self.params['phi']**2)})
        jacobian.update({'sigma_v': np.log(self.params['sigma_v'])})
        jacobian.update({'sigma_e': np.log(self.params['sigma_e'])})
        output = 0.0
        if self.no_params_to_estimate > 1:
            for param in self.params_to_estimate:
                output += jacobian[param]
        else:
            output += jacobian[self.params_to_estimate]
        return output

    generate_data = generate_data
    import_data = import_data
    store_free_params = store_free_params
    store_params = store_params
    get_free_params = get_free_params
    get_params = get_params
    create_inference_model = create_inference_model
    fix_true_params = fix_true_params
