"""Kalman methods."""
import numpy as np
from scipy.stats import norm
from state.kalman_methods.standard_filter import filter_helper
from state.kalman_methods.rts_smoother import rts_helper
from state.base_state_inference import BaseStateInference

class KalmanMethods(BaseStateInference):
    """Kalman methods."""

    def __init__(self, new_settings=None):
        self.name = None
        self.settings = {'initial_state': 0.0,
                         'initial_cov': 1e-5,
                         'estimate_gradient': False,
                         'estimate_hessian': False
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

        res = filter_helper(observations=model.obs,
                            params=(mu, phi, sigmav2, sigmae2),
                            initial_state=self.settings['initial_state'],
                            initial_cov=self.settings['initial_cov'])
        self.results.update(res)
        self.log_like = self.results['log_like']

    def smoother(self, model):
        """Kalman smoother."""
        self.name = "Kalman smoother (RTS)"
        self.filter(model)

        mu = model.params['mu']
        phi = model.params['phi']
        sigmav2 = model.params['sigma_v']**2
        sigmae2 = model.params['sigma_e']**2

        res = rts_helper(observations=model.obs,
                         params=(mu, phi, sigmav2, sigmae2),
                         pred_state_est=self.results['pred_state_est'],
                         pred_state_cov=self.results['pred_state_cov'],
                         filt_state_est=self.results['filt_state_est'],
                         filt_state_cov=self.results['filt_state_cov'],
                         kalman_gain=self.results['kalman_gain'],
                         estimate_gradient=self.settings['estimate_gradient'],
                         estimate_hessian=self.settings['estimate_hessian'])
        self.results.update(res)
        self._estimate_gradient_and_hessian(model)
