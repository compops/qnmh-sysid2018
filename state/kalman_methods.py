"""Kalman methods."""
import numpy as np
from scipy.stats import norm

class FilteringSmoothing(object):
    """Kalman methods."""

    def __init__(self, new_settings=None):
        self.name = None
        self.settings = {}
        self.settings.update({'initial_state': 0.0})
        self.settings.update({'initial_cov': 1e-5})
        self.settings.update({'estimate_gradient': False})
        self.settings.update({'estimate_hessian': False})

        self.log_like = []
        self.filt_state_est = []
        self.pred_state_est = []
        self.smo_state_est = []
        self.xtraj = []
        self.pred_state_cov = []
        self.filt_state_cov = []
        self.smo_state_cov = []
        self.kalman_gain = []

        self.gradient = []
        self.gradient_internal = []
        self.hessian_internal = []

        if new_settings:
            self.settings.update(new_settings)

    def filter(self, model):
        """Kalman filter."""
        self.name = "Kalman filter"

        self.pred_state_est = np.zeros((model.no_obs + 1))
        self.pred_state_cov = np.zeros((model.no_obs + 1))
        self.filt_state_est = np.zeros((model.no_obs))
        self.filt_state_cov = np.zeros((model.no_obs))
        self.kalman_gain = np.zeros(model.no_obs)
        self.log_like = 0.0

        self.filt_state_est[0] = self.settings['initial_state']
        self.filt_state_cov[0] = self.settings['initial_cov']

        offset = model.parameters['offset']
        phi = model.parameters['phi']
        sigmav2 = model.parameters['sigma_v']**2
        sigmae2 = model.parameters['sigma_e']**2

        for i in range(1, model.no_obs):
            # Prediction step
            self.pred_state_est[i] = offset
            self.pred_state_est[i] += phi * (self.filt_state_est[i-1] - offset)
            self.pred_state_cov[i] = phi * self.filt_state_cov[i-1] * phi
            self.pred_state_cov[i] += sigmav2

            # Correction step
            pred_obs_cov = self.pred_state_cov[i] + sigmae2
            self.kalman_gain[i] = self.pred_state_cov[i] / pred_obs_cov

            scaled_innovation = (model.obs[i] - self.pred_state_est[i])
            scaled_innovation *= self.kalman_gain[i]
            self.filt_state_est[i] = self.pred_state_est[i] + scaled_innovation
            cov_change = self.kalman_gain[i] * self.pred_state_cov[i]
            self.filt_state_cov[i] = self.pred_state_cov[i] - cov_change

            mean = self.pred_state_est[i]
            stdev = np.sqrt(pred_obs_cov)
            self.log_like += norm.logpdf(model.obs[i], mean, stdev)

    def smoother(self, model):
        """Kalman smoother."""
        self.name = "Kalman smoother (RTS)"

        smo_gain = np.zeros((model.no_obs, 1))
        smo_state_cov_twostep = np.zeros((model.no_obs, 1))
        self.smo_state_est = np.zeros((model.no_obs, 1))
        self.smo_state_cov = np.zeros((model.no_obs, 1))

        offset = model.parameters['offset']
        phi = model.parameters['phi']
        sigmav2 = model.parameters['sigma_v']**2

        # Run the preliminary Kalman filter
        self.filter(model)
        self.smo_state_cov[-1] = self.filt_state_cov[-1]
        self.smo_state_est[-1] = self.filt_state_est[-1]

        for i in range((model.no_obs - 2), 0, -1):
            smo_gain[i] = self.filt_state_cov[i] * phi
            smo_gain[i] /= self.pred_state_cov[i+1]
            diff = self.smo_state_est[i+1] - self.pred_state_est[i+1]
            self.smo_state_est[i] = self.filt_state_est[i] + smo_gain[i] * diff
            diff = self.smo_state_cov[i+1] - self.pred_state_cov[i+1]
            self.smo_state_cov[i] = self.filt_state_cov[i]
            self.smo_state_cov[i] += smo_gain[i]**2 * diff

        if self.settings['estimate_gradient']:
            # Calculate the two-step smoothing covariance
            two_step = (1 - self.kalman_gain[-1]) * phi
            two_step *= self.filt_state_cov[-1]
            smo_state_cov_twostep[model.no_obs - 1] = two_step

            for i in range((model.no_obs-2), 0, -1):
                term1 = self.filt_state_cov[i] * smo_gain[i-1]
                term2 = smo_gain[i-1]**2
                term3 = smo_state_cov_twostep[i+1]
                term4 = phi * self.filt_state_cov[i]
                smo_state_cov_twostep[i] = term1 + term2 * (term3 - term4)

            # Gradient and Hessian estimation using the Segal and Weinstein estimators
            gradient_part = np.zeros((4, model.no_obs))
            for i in range(1, model.no_obs):
                next_state = self.smo_state_est[i]
                cur_state = self.smo_state_est[i-1]
                eta = next_state * next_state + self.smo_state_cov[i]
                eta1 = cur_state**2 + self.smo_state_cov[i-1]
                psi = cur_state * next_state + smo_state_cov_twostep[i]
                quad_term = next_state - offset - phi * (cur_state - offset)
                sigmav2 = model.parameters['sigma_v']**(-2)

                gradient_part[0, i] = sigmav2 * quad_term * (1.0 - phi)

                term1 = sigmav2 * (1.0 - phi**2)
                term2 = psi - phi * eta1
                term2 -= cur_state * offset * (1.0 - 2.0 * phi)
                term2 += -next_state * offset + offset**2 * (1.0 - phi)
                gradient_part[1, i] = term1 * term2

                term1 = eta - 2 * phi * psi + phi**2 * eta1
                term2 = -2.0 * (next_state - phi * self.smo_state_est[i-1])
                term2 *= (1.0 - phi) * offset
                term3 = offset**2 * (1.0 - phi)**2
                gradient_part[2, i] = sigmav2 * (term1 + term2 + term3) - 1.0
                gradient_part[3, i] = 0.0

            gradient_sum = np.sum(gradient_part, axis=1)

            gradient = {}
            gradient_internal = []
            i = 0
            for parameter in model.parameters.keys():
                if parameter in model.params_to_estimate:
                    gradient.update({parameter: gradient_sum[i]})
                    gradient_internal.append(gradient_sum[i])
                i += 1

        if self.settings['estimate_hessian']:
            part1 = np.mat(gradient_part).transpose()
            part1 = np.dot(np.mat(gradient_part), part1)
            part2 = np.mat(gradient_sum)
            part2 = np.dot(np.mat(gradient_sum).transpose(), part2)
            hessian = part1 - part2 / model.no_obs

        # Add the log-prior derivatives
        log_prior_gradient = model.log_prior_gradient()
        log_prior_hessian = model.log_prior_hessian()
        i = 0
        for first_param in log_prior_gradient.keys():
            gradient_sum[i] += log_prior_gradient[first_param]
            if self.settings['estimate_hessian']:
                j = 0
                for second_param in log_prior_gradient.keys():
                    hessian[i, j] -= log_prior_hessian[first_param]
                    hessian[i, j] -= log_prior_hessian[second_param]
                    j += 1
            i += 1

        if self.settings['estimate_gradient']:
            self.gradient_internal = np.array(gradient_internal)
            self.gradient = gradient

        if self.settings['estimate_hessian']:
            self.gradient_internal = np.array(gradient_internal)
            self.gradient = gradient
            idx = model.params_to_estimate_idx
            self.hessian_internal = hessian[np.ix_(idx, idx)]
