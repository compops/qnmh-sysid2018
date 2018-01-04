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

"""Defines the basic model class."""
import numpy as np

from helpers.data_handling import generate_data, import_data
from helpers.data_handling import import_data_quandl
from helpers.inference_model import create_inference_model, fix_true_params
from helpers.model_params import store_free_params, store_params
from helpers.model_params import get_free_params, get_params, get_all_params


class BaseModel(object):

    name = None
    file_prefix = None

    params = {}
    free_params = {}
    no_params = 0
    params_prior = {}
    intial_state = []
    no_obs = []

    states = []
    inputs = []
    obs = []

    params_to_estimate_idx = []
    no_params_to_estimate = 0
    params_to_estimate = []
    true_params = []

    def __init__(self):
        pass

    def __repr__(self):
        print("===============================================================")
        print("===============================================================")
        print("Model: " + self.name)
        print("")
        print("Standard parameters:")
        print("--------------------------------------------")
        for param in self.params:
            print(param + ": {}".format(self.params[param]) )
        print("")
        if self.free_params:
            print("Free (unrestricted) parameters:")
            print("--------------------------------------------")
            for param in self.free_params:
                print(param + ": {}".format(self.free_params[param]))
            print("")
        if len(self.params_to_estimate) > 0:
            print("Inference on the following parameters: ")
            print("--------------------------------------------")
            for param in self.params_to_estimate:
                print(param)
            print("")
            print("Parameter priors:")
            print("--------------------------------------------")
            for param in self.params_prior:
                param_prior = self.params_prior[param]
                print(param + " has prior {} with hyperparameters: {} and {}. \
                ".format(param_prior[0].__name__,
                         param_prior[1],
                         param_prior[2]))
            print("")
        print("Data contents:")
        print("--------------------------------------------")
        if isinstance(self.obs, np.ndarray):
            print("Model contains {} observations.".format(self.no_obs))
        if isinstance(self.states, np.ndarray):
            print("Model contains {} states.".format(len(self.states)))
        if isinstance(self.inputs, np.ndarray):
            print("Model contains {} inputs.".format(len(self.inputs)))
        print("===============================================================")
        print("===============================================================")
        return " "

    def generate_initial_state(self, no_samples):
        """ Generates no_samples from the initial state distribution.

            Args:
                no_samples: number of samples to generate (integer).

            Returns:
                An array with no_samples from the initial state distribution.

        """
        raise NotImplementedError

    def generate_state(self, cur_state, time_step):
        """ Generates a new state by the state dynamics.

            Args:
                cur_state: the current state (array).
                time_step: the current time step (integer).

            Returns:
                An array of samples from the next time step.

        """
        raise NotImplementedError

    def evaluate_state(self, next_state, cur_state, time_step):
        """ Computes the probability of a state transition.

            Args:
                next_state: the next state (array)
                cur_state: the current state (array).
                time_step: the current time step (integer).

            Returns:
                An array of transition log-probabilities.

        """
        raise NotImplementedError

    def generate_obs(self, cur_state, time_step):
        """ Generates a new observation by the observation dynamics.

            Args:
                cur_state: the current state (array).
                time_step: the current time step (integer).

            Returns:
                An array of observations.

        """
        raise NotImplementedError

    def evaluate_obs(self, cur_state, time_step):
        """ Computes the probability of obtaining an observation.

            Args:
                cur_state: the current state (array).
                time_step: the current time step (integer).

            Returns:
                An array of observation log-probabilities.

        """
        raise NotImplementedError

    def check_parameters(self):
        """" Checks if parameters satisfies hard constraints on the parameters.

                Returns:
                    Boolean to indicate if the current parameters results in
                    a stable system and obey the constraints on their values.

        """
        raise NotImplementedError

    def log_prior(self):
        """ Returns the logarithm of the prior distribution.

            Returns:
                First value: a dict with an entry for each parameter.
                Second value: the sum of the log-prior for all variables.

        """
        prior = {}
        for param in self.params:
            dist = self.params_prior[param][0]
            hyppar1 = self.params_prior[param][1]
            hyppar2 = self.params_prior[param][2]
            prior.update(
                {param: dist.logpdf(self.params[param], hyppar1, hyppar2)})

        prior_sum = 0.0
        for param in self.params_to_estimate:
            prior_sum += prior[param]

        return prior, prior_sum

    def log_prior_gradient(self):
        """ The gradient of the logarithm of the prior.

            Returns:
                A dict with an entry for each parameter.

        """
        gradients = {}
        for param in self.params:
            dist = self.params_prior[param][0]
            value = self.params[param]
            hyppar1 = self.params_prior[param][1]
            hyppar2 = self.params_prior[param][2]
            gradient = dist.logpdf_gradient(value, hyppar1, hyppar2)
            gradients.update({param: gradient})

        return gradients

    def log_prior_hessian(self):
        """ The Hessian of the logarithm of the prior.

            Returns:
                A dict with an entry for each parameter.

        """
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

        raise NotImplementedError

    def transform_params_to_free(self):
        """ Computes and store the values of the reparameterised parameters.

            These transformations are dictated directly from the model. See
            the docstring for the model class for more information. The
            values of the reparameterised parameters are computed by applying
            the transformation to the current standard parameters stored in
            the model object.

        """
        raise NotImplementedError

    def transform_params_from_free(self):
        """ Computes and store the values of the standard parameters.

             These transformations are dictated directly from the model. See
             the docstring for the model class for more information. The
             values of the standard parameters are computed by applying
             the transformation to the current reparameterised parameters stored
             in the model object.

        """
        raise NotImplementedError

    def _compile_log_jacobian(self, jacobian):
        output = 0.0
        if self.no_params_to_estimate > 1:
            for param in self.params_to_estimate:
                output += jacobian[param]
        else:
            output += jacobian[self.params_to_estimate]
        return output

    def log_jacobian(self):
        """ Computes the sum of the log-Jacobian.

            These Jacobians are dictated by the transformations for the model.
            See the docstring for the model class for more information.

            Returns:
                the sum of the logarithm of the Jacobian of the parameter
                transformation for the parameters under inference as listed
                in params_to_estimate.

        """
        raise NotImplementedError

    # Helpers for generating and importing data
    generate_data = generate_data
    import_data = import_data
    import_data_quandl = import_data_quandl

    # Helpers for handling parameters
    store_free_params = store_free_params
    store_params = store_params
    get_free_params = get_free_params
    get_params = get_params
    get_all_params = get_all_params
    fix_true_params = fix_true_params

    # Helpers for doing inference
    create_inference_model = create_inference_model
