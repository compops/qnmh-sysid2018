"""Kalman methods."""
import numpy as np
from scipy.stats import norm
from state.kalman_methods.standard_filter import filter_helper
from state.kalman_methods.rts_smoother import rts_helper

class KalmanMethods(object):
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
        mu = model.params['mu']
        phi = model.params['phi']
        sigmav2 = model.params['sigma_v']**2
        sigmae2 = model.params['sigma_e']**2

        output = filter_helper(observations=model.obs,
                               params=(mu, phi, sigmav2, sigmae2),
                               initial_state=self.settings['initial_state'],
                               initial_cov=self.settings['initial_cov'])

        self.pred_state_est = output['pred_state_est']
        self.pred_state_cov = output['pred_state_cov']
        self.kalman_gain = output['kalman_gain']
        self.filt_state_est = output['filt_state_est']
        self.filt_state_cov = output['filt_state_cov']
        self.log_like = output['log_like']

    def smoother(self, model):
        """Kalman smoother."""
        self.name = "Kalman smoother (RTS)"
        self.filter(model)

        mu = model.params['mu']
        phi = model.params['phi']
        sigmav2 = model.params['sigma_v']**2
        sigmae2 = model.params['sigma_e']**2

        output = rts_helper(observations=model.obs,
                            params=(mu, phi, sigmav2, sigmae2),
                            pred_state_est=self.pred_state_est,
                            pred_state_cov=self.pred_state_cov,
                            filt_state_est=self.filt_state_est,
                            filt_state_cov=self.filt_state_cov,
                            kalman_gain=self.kalman_gain,
                            estimate_gradient=self.settings['estimate_gradient'],
                            estimate_hessian=self.settings['estimate_hessian']
                           )

        gradient_sum = output['gradient_sum']
        gradient_part = output['gradient_part']
        hessian = output['hessian']
        self.smo_state_est = output['smo_state_est']
        self.smo_state_cov = output['smo_state_cov']

        gradient = {}
        gradient_internal = []
        i = 0
        for parameter in model.params.keys():
            if parameter in model.params_to_estimate:
                gradient.update({parameter: gradient_sum[i]})
                gradient_internal.append(gradient_sum[i])
            i += 1

        # Add the log-prior derivatives
        log_prior_gradient = model.log_prior_gradient()
        log_prior_hessian = model.log_prior_hessian()
        i = 0
        for first_param in log_prior_gradient.keys():
            if self.settings['estimate_gradient']:
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
            self.hessian_internal = np.array(hessian[np.ix_(idx, idx)])
