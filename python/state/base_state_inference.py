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
        self.name

    def _estimate_gradient_and_hessian(self, model):
        """Inserts gradients and Hessian of the log-priors into the estimates
        of the gradient and Hessian of the log-likelihood."""

        gradient_estimate = self.results['log_joint_gradient_estimate']
        hessian_estimate = self.results['log_joint_hessian_estimate']
        gradient = {}
        gradient_internal = []

        i = 0
        for param in model.params.keys():
            if param in model.params_to_estimate:
                gradient.update({param: gradient_estimate[i]})
                gradient_internal.append(gradient_estimate[i])
            i += 1

        # Add the log-prior derivatives
        if self.settings['estimate_gradient']:
            log_prior_gradient = model.log_prior_gradient()
            i = 0

            for param1 in log_prior_gradient.keys():
                gradient_estimate[i] += log_prior_gradient[param1]
                log_prior_hessian = model.log_prior_hessian()
                j = 0

                for param2 in log_prior_gradient.keys():
                    hessian_estimate[i, j] -= log_prior_hessian[param1]
                    hessian_estimate[i, j] -= log_prior_hessian[param2]
                    j += 1
                i += 1
            self.results.update({'gradient_internal': np.array(gradient_internal)})
            self.results.update({'gradient': gradient})
            idx = model.params_to_estimate_idx
            self.results.update({'hessian_internal': np.array(hessian_estimate[np.ix_(idx, idx)])})
