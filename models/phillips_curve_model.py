"""System model class for a non-linear Phillips curve model."""
import numpy as np
from scipy.stats import norm

from helpers.data_generation import generate_data, import_data
from helpers.inference_model import create_inference_model, fix_true_params
from helpers.model_params import store_free_params, store_params
from helpers.model_params import get_free_params, get_params, get_all_params
from helpers.distributions import normal
from helpers.distributions import gamma

class SystemModel(object):
    """System model class for a non-linear Phillips curve model."""

    def __init__(self):
        self.name = "Phillips curve model with rational expectations."
        self.file_prefix = "phillips"

        self.params = {'alpha': 0.2, 'phi': 0.75, 'beta': 0.0, 'sigma_e': 0.1}
        self.free_params = {'alpha': 0.2, 'phi': 0.64, 'beta': 0.0,
                            'sigma_e': -2.3}
        self.no_params = len(self.params)
        self.params_prior = {'alpha': (normal, 0.5, 0.2),
                             'phi': (normal, 0.8, 0.1),
                             'beta': (normal, 0.0, 0.1),
                             'sigma_e': (gamma, 2.0, 4.0)
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
        """Generates no_samples from the initial state distribution."""
        return 2.0 + 2.0 * np.random.normal(size=(1, no_samples))

    def generate_state(self, cur_state, time_step):
        """Generates a new state by the state dynamics given cur_state."""
        mean = self.params['phi'] * cur_state
        mean += self.params['alpha'] / (1.0 + np.exp(-cur_state))
        noise_stdev = 1.0 + np.exp(-np.abs(self.inputs[time_step] - cur_state))
        return np.abs(mean + noise_stdev**(-1) * np.random.randn(1, len(cur_state)))

    def generate_obs(self, cur_state, time_step):
        """Generates a new observation by the observation dynamics
           given cur_state."""
        mean = self.obs[time_step-1]
        mean += self.params['beta'] * (self.inputs[time_step] - cur_state)
        noise_stdev = self.params['sigma_e']
        return mean + noise_stdev * np.random.randn(1, len(cur_state))

    def evaluate_obs(self, cur_state, time_step):
        """Evaluates the probability of cur_state and cur_obs
           under the observation model."""
        cur_obs = self.obs[time_step]
        mean = self.params['beta'] * (self.inputs[time_step] - cur_state)
        mean += self.obs[time_step-1]
        noise_stdev = self.params['sigma_e']
        return norm.logpdf(cur_obs, mean, noise_stdev)

    def check_parameters(self):
        """"Checks if parameters satisifes hard constraints."""
        if np.abs(self.params['phi']) > 1.0:
            parameters_are_okey = False
        elif self.params['sigma_e'] < 0.0:
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
        gradients['sigma_e'] *= self.params['sigma_e']
        hessians['sigma_e'] *= self.params['sigma_e']
        hessians['sigma_e'] += gradients['sigma_e']

        return hessians

    def log_joint_gradient(self, next_state, cur_state, time_index):
        """Returns the gradient of the logarithm of the joint distributions of
        states and observations as a dictionary with a member for each
        parameter."""
        prev_obs = self.obs[time_index-1]
        cur_obs = self.obs[time_index]
        cur_input = self.obs[time_index]

        alpha_term = (1.0 + np.exp(-cur_state))**(-1)
        beta_term = cur_input - cur_state

        state_quad_term = next_state - self.params['phi'] * cur_state
        state_quad_term -= self.params['alpha'] * alpha_term

        obs_quad_term = cur_obs - prev_obs - self.params['beta'] * beta_term

        q_matrix = (1.0 + np.exp(-np.abs(cur_obs - cur_state)))**(-2)
        r_matrix = self.params['sigma_e']**(-2)

        gradient_alpha = q_matrix * state_quad_term * alpha_term
        gradient_phi = q_matrix * state_quad_term * cur_state
        gradient_phi *= (1.0 - self.params['phi']**2)
        gradient_beta = r_matrix * obs_quad_term * beta_term
        gradient_sigmae = r_matrix * obs_quad_term**2 - 1.0

        gradient = {}
        gradient.update({'alpha': gradient_alpha})
        gradient.update({'phi': gradient_phi})
        gradient.update({'beta': gradient_beta})
        gradient.update({'sigma_e': gradient_sigmae})
        return gradient

    def transform_params_to_free(self):
        """Transforms the current parameters to their free parameterisation."""
        self.free_params['alpha'] = self.params['alpha']
        self.free_params['phi'] = np.arctanh(self.params['phi'])
        self.free_params['beta'] = self.params['beta']
        self.free_params['sigma_e'] = np.log(self.params['sigma_e'])

    def transform_params_from_free(self):
        """Get the model parameters by transforming from their free
        parameterisation."""
        self.params['alpha'] = self.free_params['alpha']
        self.params['phi'] = np.tanh(self.free_params['phi'])
        self.params['beta'] = self.free_params['beta']
        self.params['sigma_e'] = np.exp(self.free_params['sigma_e'])

    def log_jacobian(self):
        """Returns the sum of the log-Jacobian for all parameters to be
        estimated."""
        jacobian = {}
        jacobian.update({'alpha': 0.0})
        jacobian.update({'phi': np.log(1.0 - self.params['phi']**2)})
        jacobian.update({'beta': 0.0})
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
    get_all_params = get_all_params
    create_inference_model = create_inference_model
    fix_true_params = fix_true_params
