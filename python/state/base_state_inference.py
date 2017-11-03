"""The base state inference object."""

import numpy as np

class BaseStateInference(object):
    name = []
    settings = {}
    results = {}
    model = {}

    no_obs = 0
    log_like = []
    gradient = []
    gradient_internal = []
    hessian_internal = []

    def __repr__(self):
        pass

    def _estimate_gradient_and_hessian(self, model):
        """Inserts gradients and Hessian of the log-priors into the estimates
        of the gradient and Hessian of the log-likelihood."""

        gradient_estimate = self.results['log_joint_gradient_estimate']
        hessian_estimate = self.results['log_joint_hessian_estimate']

        gradient = {}
        gradient_internal = []
        i = 0
        for parameter in model.params.keys():
            if parameter in model.params_to_estimate:
                gradient.update({parameter: gradient_estimate[i]})
                gradient_internal.append(gradient_estimate[i])
            i += 1

        # Add the log-prior derivatives
        if self.settings['estimate_gradient']:
            log_prior_gradient = model.log_prior_gradient()
            i = 0

            for first_param in log_prior_gradient.keys():
                gradient_estimate[i] += log_prior_gradient[first_param]

                if self.settings['estimate_hessian']:
                    log_prior_hessian = model.log_prior_hessian()
                    j = 0

                    for second_param in log_prior_gradient.keys():
                        hessian_estimate[i, j] -= log_prior_hessian[first_param]
                        hessian_estimate[i, j] -= log_prior_hessian[second_param]
                        j += 1
                i += 1

        if self.settings['estimate_gradient']:
            self.gradient_internal = np.array(gradient_internal)
            self.gradient = gradient

        if self.settings['estimate_hessian']:
            idx = model.params_to_estimate_idx
            self.hessian_internal = np.array(hessian_estimate[np.ix_(idx, idx)])
