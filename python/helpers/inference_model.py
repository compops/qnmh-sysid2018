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

"""Helpers for creating inference models from system models."""
import copy
import numpy as np

def create_inference_model(model, params_to_estimate):
    """ Transforms a model object into an inference object.

        Adds additional information into a system model to enable it to be
        used for inference. This information includes the parameters to
        estimate in the model.

        Args:
            model: model object
            params_to_estimate: list of parameters to estimate. For example
                                params_to_estimate = ('mu', 'phi').

        Returns:
           Nothing.

    """

    model.model_type = "Inference model"

    if isinstance(params_to_estimate, str):
        model.no_params_to_estimate = 1
    else:
        model.no_params_to_estimate = len(params_to_estimate)

    model.params_to_estimate = params_to_estimate
    model.params_to_estimate_idx = []

    for param in model.params:
        if param in params_to_estimate:
            new_param = params_to_estimate.index(param)
            model.params_to_estimate_idx.append(int(new_param))
    model.params_to_estimate_idx = np.asarray(model.params_to_estimate_idx)

def fix_true_params(model):
    """ Creates a copy of the true parameters into the model object. """
    model.true_params = copy.deepcopy(model.params)