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

"""Helpers for storing and returning parameters in inference/system models."""

import copy
import numpy as np

def store_free_params(model, new_params):
    """ Stores reparameterised parameters to the model.

        Stores new reparameterised (unrestricted) values of the parameters
        to the model object. The restricted parameters are updated accordingly.
        Only the parameters used for inference are required. The remaining
        parameters are copied from the trueParams attribute.

        Args:
            model: model object
            new_params: an array with the new reparameterised parameters. The
                        order must be as in the list params_to_estimate.

        Returns:
           Nothing.

    """
    model.params = copy.deepcopy(model.true_params)
    model.transform_params_to_free()

    if isinstance(new_params, float) or (isinstance(new_params, np.ndarray) and
                                         len(new_params) == 1):
        model.free_params[model.params_to_estimate] = float(new_params)
    else:
        for param in model.params_to_estimate:
            param_single = new_params[model.params_to_estimate.index(param)]
            model.free_params[param] = float(param_single)
    model.transform_params_from_free()

def store_params(model, new_params):
    """ Stores (restricted) parameters to the model.

        Stores new values of the parameters to the model object. The
        reparameterised (unrestricted) parameters are updated accordingly.
        Only the parameters used for inference are required. The remaining
        parameters are copied from the trueParams attribute.

        Args:
            model: model object
            new_params: an array with the new parameters. The order must be
                        the same as in the list params_to_estimate.

        Returns:
           Nothing.

    """
    model.params = copy.deepcopy(model.true_params)

    if isinstance(new_params, float) or (isinstance(new_params, np.ndarray) and
                                         len(new_params) == 1):
        model.params[model.params_to_estimate] = float(new_params)
    else:
        for param in model.params_to_estimate:
            idx = model.params_to_estimate.index(param)
            param_single = float(new_params[idx])
            model.params[param] = param_single
    model.transform_params_to_free()

def get_free_params(model):
    """ Returns the reparameterised parameters under inference in the model.

        Args:
            model: model object

        Returns:
           An array with the current reparameterised values for the parameters
           under inference in the model. The order is the same as in the list
           params_to_estimate.

    """
    parameters = []
    if isinstance(model.params_to_estimate, str):
        parameters.append(model.free_params[model.params_to_estimate])
    else:
        for param in model.params_to_estimate:
            parameters.append(model.free_params[param])
    return np.array(parameters)

def get_params(model):
    """ Returns the parameters under inference in the model.

        Args:
            model: model object

        Returns:
           An array with the current values for the parameters under inference
           in the model. The order is the same as in the list params_to_estimate.

    """
    parameters = []
    if isinstance(model.params_to_estimate, str):
        parameters.append(model.params[model.params_to_estimate])
    else:
        for param in model.params_to_estimate:
            parameters.append(model.params[param])
    return np.array(parameters)

def get_all_params(model):
    """ Returns all the parameters in the model.

        Args:
            model: model object

        Returns:
           An array with the current values of all parameters in the model.

    """
    parameters = []
    for param in model.params:
        parameters.append(model.params[param])
    return np.array(parameters)