"""Kalman methods using Cython."""
import numpy as np

from state.kalman_methods.cython_helper import filter, smoother
from state.base_state_inference import BaseStateInference

class KalmanMethodsCython(BaseStateInference):
    """Kalman methods."""

    def __init__(self, new_settings=None):
        self.name = None
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

        res = filter(observations=model.obs,
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

        res = smoother(observations=model.obs,
                       params=(mu, phi, sigmav2, sigmae2),
                       pred_state_est=self.results['pred_state_est'],
                       pred_state_cov=self.results['pred_state_cov'],
                       filt_state_est=self.results['filt_state_est'],
                       filt_state_cov=self.results['filt_state_cov'],
                       kalman_gain=self.results['kalman_gain'],
                       estimate_gradient=self.settings['estimate_gradient'])
        self.results.update(res)
        self._estimate_gradient_and_hessian(model)
