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

"""Kalman methods."""
import numpy as np
from scipy.stats import norm
from state.base_state_inference import BaseStateInference


class KalmanMethods(BaseStateInference):
    """Kalman methods."""

    def __init__(self, new_settings=None):
        self.name = "Kalman methods"
        self.settings = {'initial_state': 0.0,
                         'initial_cov': 1e-5,
                         'estimate_gradient': False
                         }
        if new_settings:
            self.settings.update(new_settings)

    def filter(self, model):
        """Kalman filter."""
        self.name = "Kalman filter"

        mu = model.params['mu']
        phi = model.params['phi']
        sigmav2 = model.params['sigma_v']**2
        sigmae2 = model.params['sigma_e']**2

        pred_state_est = np.zeros((model.no_obs + 1))
        pred_state_cov = np.zeros((model.no_obs + 1))
        filt_state_est = np.zeros((model.no_obs + 1))
        filt_state_cov = np.zeros((model.no_obs + 1))
        kalman_gain = np.zeros(model.no_obs + 1)
        log_like = 0.0

        filt_state_est[0] = self.settings['initial_state']
        filt_state_cov[0] = self.settings['initial_cov']

        for i in range(1, model.no_obs + 1):
            # Prediction step
            pred_state_est[i] = mu
            pred_state_est[i] += phi * (filt_state_est[i - 1] - mu)
            pred_state_cov[i] = phi * filt_state_cov[i - 1] * phi
            pred_state_cov[i] += sigmav2

            # Correction step
            pred_obs_cov = pred_state_cov[i] + sigmae2
            kalman_gain[i] = pred_state_cov[i] / pred_obs_cov

            scaled_innovation = (model.obs[i] - pred_state_est[i])
            scaled_innovation *= kalman_gain[i]
            filt_state_est[i] = pred_state_est[i] + scaled_innovation
            cov_change = kalman_gain[i] * pred_state_cov[i]
            filt_state_cov[i] = pred_state_cov[i] - cov_change

            mean = pred_state_est[i]
            stdev = np.sqrt(pred_obs_cov)
            log_like += norm.logpdf(model.obs[i], mean, stdev)

        self.results.update({'pred_state_est': pred_state_est,
                             'pred_state_cov': pred_state_cov,
                             'kalman_gain': kalman_gain,
                             'filt_state_est': filt_state_est,
                             'filt_state_cov': filt_state_cov,
                             'log_like': log_like,
                             'state_trajectory': np.zeros(model.no_obs+1)
                             })

    def smoother(self, model):
        """Kalman smoother."""
        self.name = "Kalman smoother (RTS)"
        self.filter(model)

        mu = model.params['mu']
        phi = model.params['phi']
        sigmav2 = model.params['sigma_v']**2
        sigmae2 = model.params['sigma_e']**2

        pred_state_est = self.results['pred_state_est']
        pred_state_cov = self.results['pred_state_cov']
        filt_state_est = self.results['filt_state_est']
        filt_state_cov = self.results['filt_state_cov']
        kalman_gain = self.results['kalman_gain']

        smo_gain = np.zeros((model.no_obs + 1, 1))
        smo_state_cov_twostep = np.zeros((model.no_obs + 1, 1))
        smo_state_est = np.zeros((model.no_obs + 1, 1))
        smo_state_cov = np.zeros((model.no_obs + 1, 1))
        gradient_part = []

        log_joint_gradient_estimate = []
        log_joint_hessian_estimate = []

        # Run the preliminary Kalman filter
        smo_state_est[-1] = filt_state_est[-1]
        smo_state_cov[-1] = filt_state_cov[-1]

        for i in range((model.no_obs - 1), 0, -1):
            smo_gain[i] = filt_state_cov[i] * phi
            smo_gain[i] /= pred_state_cov[i + 1]
            diff = smo_state_est[i + 1] - pred_state_est[i + 1]
            smo_state_est[i] = filt_state_est[i] + smo_gain[i] * diff
            diff = smo_state_cov[i + 1] - pred_state_cov[i + 1]
            smo_state_cov[i] = filt_state_cov[i]
            smo_state_cov[i] += smo_gain[i]**2 * diff

        if self.settings['estimate_gradient']:
            # Calculate the two-step smoothing covariance
            two_step = (1 - kalman_gain[-1]) * phi
            two_step *= filt_state_cov[-1]
            smo_state_cov_twostep[model.no_obs - 1] = two_step

            for i in range((model.no_obs - 1), 0, -1):
                term1 = filt_state_cov[i] * smo_gain[i - 1]
                term2 = smo_gain[i - 1]**2
                term3 = smo_state_cov_twostep[i + 1]
                term4 = phi * filt_state_cov[i]
                smo_state_cov_twostep[i] = term1 + term2 * (term3 - term4)

        if self.settings['estimate_gradient']:
            # Gradient and Hessian estimation using Segal-Weinstein estimator
            gradient_part = np.zeros((4, model.no_obs))
            for i in range(1, model.no_obs):
                next_state = smo_state_est[i]
                cur_state = smo_state_est[i - 1]
                eta = next_state * next_state + smo_state_cov[i]
                eta1 = cur_state**2 + smo_state_cov[i - 1]
                psi = cur_state * next_state + smo_state_cov_twostep[i]
                quad_term = next_state - mu - phi * (cur_state - mu)
                isigmav2 = 1.0 / sigmav2

                gradient_part[0, i] = isigmav2 * quad_term * (1.0 - phi)

                term1 = isigmav2 * (1.0 - phi**2)
                term2 = psi - phi * eta1
                term2 -= cur_state * mu * (1.0 - 2.0 * phi)
                term2 += -next_state * mu + mu**2 * (1.0 - phi)
                gradient_part[1, i] = term1 * term2

                term1 = eta - 2 * phi * psi + phi**2 * eta1
                term2 = -2.0 * (next_state - phi * smo_state_est[i - 1])
                term2 *= (1.0 - phi) * mu
                term3 = mu**2 * (1.0 - phi)**2
                gradient_part[2, i] = isigmav2 * (term1 + term2 + term3) - 1.0
                gradient_part[3, i] = 0.0

            log_joint_gradient_estimate = np.sum(gradient_part, axis=1)

            part1 = np.mat(gradient_part).transpose()
            part1 = np.dot(np.mat(gradient_part), part1)
            part2 = np.mat(log_joint_gradient_estimate)
            part2 = np.dot(np.mat(log_joint_gradient_estimate).transpose(), part2)
            log_joint_hessian_estimate = part1 - part2 / model.no_obs

        self.results.update({'smo_state_cov': smo_state_cov,
                             'smo_state_est': smo_state_est,
                             'log_joint_gradient_estimate': log_joint_gradient_estimate,
                             'log_joint_hessian_estimate': log_joint_hessian_estimate
                             })
        if self.settings['estimate_gradient']:
            self._estimate_gradient_and_hessian(model)
