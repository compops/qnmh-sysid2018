"""The base parameter inference object."""

import numpy as np

class BaseParameterInference(object):
    name = {}
    settings = {}
    results = {}
    alg_type = []
    model = {}

    no_iters = 0
    no_params_to_estimate = 0
    no_obs = 0

    def __repr__(self):
        pass