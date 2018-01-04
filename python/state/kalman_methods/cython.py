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

"""Kalman methods using Cython."""
import numpy as np

from state.kalman_methods.cython_helper import kf_filter, rts_smoother
from state.base_state_inference import BaseStateInference

class KalmanMethodsCython(BaseStateInference):
    """Kalman methods."""

    def __init__(self, new_settings=None):
        self.name = "Kalman methods (Cython implementation)"
        self.settings = {'initial_state': 0.0,
                         'initial_cov': 1e-5,
                         'estimate_gradient': False
                         }
        if new_settings:
            self.settings.update(new_settings)

    def filter(self, model):
        """Kalman filter."""
        self.name = "Kalman filter"
        obs = np.array(model.obs.flatten())
        params = model.get_all_params()
        xhatp, Pp, xhatf, Pf, ll = kf_filter(obs, mu=params[0], phi=params[1],
                                             sigmav=params[2], sigmae=params[3],
                                             initial_state=self.settings['initial_state'],
                                             initial_cov=self.settings['initial_cov'])

        self.results.update({'pred_state_est': np.array(xhatp).reshape((model.no_obs+1, 1))})
        self.results.update({'pred_state_cov': np.array(Pp).reshape((model.no_obs+1, 1))})
        self.results.update({'filt_state_est': np.array(xhatf).reshape((model.no_obs+1, 1))})
        self.results.update({'filt_state_cov': np.array(Pf).reshape((model.no_obs+1, 1))})
        self.results.update({'log_like': float(ll)})
        self.results.update({'state_trajectory': np.zeros(model.no_obs+1)})

    def smoother(self, model):
        """Kalman smoother."""
        self.name = "Kalman smoother (RTS)"
        obs = np.array(model.obs.flatten())
        params = model.get_all_params()
        xhatp, Pp, xhatf, Pf, ll, xhats, Ps, grad = rts_smoother(obs,
                                                                 mu=params[0],
                                                                 phi=params[1],
                                                                 sigmav=params[2],
                                                                 sigmae=params[3],
                                                                 initial_state=self.settings['initial_state'],
                                                                 initial_cov=self.settings['initial_cov'])


        # Compute estimate of gradient and Hessian
        grad = np.array(grad).reshape((4, model.no_obs+1))
        log_joint_gradient_estimate = np.sum(grad, axis=1)

        part1 = np.mat(grad).transpose()
        part1 = np.dot(np.mat(grad), part1)
        part2 = np.mat(log_joint_gradient_estimate)
        part2 = np.dot(np.mat(log_joint_gradient_estimate).transpose(), part2)

        try:
            log_joint_hessian_estimate = part1 - part2 / model.no_obs
        except RuntimeWarning:
            print(part1)
            print(part2)
            log_joint_hessian_estimate = np.eye(model.no_params_to_estimate)

        self.results.update({'pred_state_est': np.array(xhatp).reshape((model.no_obs+1, 1))})
        self.results.update({'pred_state_cov': np.array(Pp).reshape((model.no_obs+1, 1))})
        self.results.update({'filt_state_est': np.array(xhatf).reshape((model.no_obs+1, 1))})
        self.results.update({'filt_state_cov': np.array(Pf).reshape((model.no_obs+1, 1))})
        self.results.update({'smo_state_est': np.array(xhats).reshape((model.no_obs+1, 1))})
        self.results.update({'smo_state_cov': np.array(Ps).reshape((model.no_obs+1, 1))})
        self.results.update({'log_like': float(ll)})
        self.results.update({'log_joint_gradient_estimate': log_joint_gradient_estimate})
        self.results.update({'log_joint_hessian_estimate': log_joint_hessian_estimate})
        self.results.update({'state_trajectory': np.zeros(model.no_obs+1)})

        self._estimate_gradient_and_hessian(model)
