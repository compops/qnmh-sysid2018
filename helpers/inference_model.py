"""Helpers for creating inference models from system models."""
import copy
import numpy as np

def create_inference_model(model, params_to_estimate):
    """Transoforms a system model into a system model to estimate
    params_to_estimate in a later stage with an inference algorithm."""

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
    model.true_params = copy.deepcopy(model.params)