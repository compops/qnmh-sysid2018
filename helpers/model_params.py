"""Helpers for storing and returning parameters in inference/system models."""

import copy
import numpy as np

def store_free_params(model, new_params):
    """Stores new unrestricted (reparameterised) values of the parameters
    to the model object. The restricted parameters are updated accordingly."""
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
    """Stores new restricted (original) values of the parameters
    to the model object. The unrestricted parameters are updated accordingly."""
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
    """Returns the unrestricted values of the model parameters as a vector."""
    parameters = []
    if isinstance(model.params_to_estimate, str):
        parameters.append(model.free_params[model.params_to_estimate])
    else:
        for param in model.params_to_estimate:
            parameters.append(model.free_params[param])
    return np.array(parameters)

def get_params(model):
    """Returns the restricted values of the model parameters as a vector."""
    parameters = []
    if isinstance(model.params_to_estimate, str):
        parameters.append(model.params[model.params_to_estimate])
    else:
        for param in model.params_to_estimate:
            parameters.append(model.params[param])
    return np.array(parameters)
