"""Helpers for creating inference models from system models."""
import copy

def get_inference_model(old_model, params_to_estimate):
    """Creates an inference model from a system model to estimate
    params_to_estimate in a later stage with an inference algorithm."""
    new_model = copy.deepcopy(old_model)
    new_model.model_type = "Inference model"

    if isinstance(params_to_estimate, str):
        new_model.no_params_to_estimate = 1
    else:
        new_model.no_params_to_estimate = len(params_to_estimate)

    new_model.params_to_estimate = params_to_estimate
    new_model.true_params = copy.deepcopy(old_model.params)
    new_model.params_to_estimate_idx = []

    for param in new_model.parameters.keys():
        if param in params_to_estimate:
            new_param = params_to_estimate.index(param)
            new_model.params_to_estimate_idx.append(new_param)

    return new_model
