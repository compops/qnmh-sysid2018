"""Particle methods."""
import numpy as np
from state.particle_methods.cython_helper import bpf_lgss
from state.particle_methods.cython_helper import flps_lgss
from state.base_state_inference import BaseStateInference

class ParticleMethods(BaseStateInference):
    """Particle methods."""

    def __init__(self, new_settings=None):

        self.settings = {'no_particles': 100,
                         'resampling_method': 'systematic',
                         'fixed_lag': 0,
                         'initial_state': 0.0,
                         'generate_initial_state': False,
                         'estimate_gradient': False,
                         'estimate_hessian': False
                         }
        if new_settings:
            self.settings.update(new_settings)

    def filter(self, model):
        """Bootstrap particle filter for linear Gaussian model."""
        self.name = "Bootstrap particle filter (Cython)"
        res = bpf_lgss(np.array(model.obs, dtype=np.float64),
                       params=model.get_all_params(),
                       no_particles=self.settings['no_particles'])
        self.results.update(res)

    def smoother(self, model):
        """Fixed-lag particle smoother for linear Gaussian model."""
        self.name = "Fixed-lag particle smoother (Cython)"
        self.bpf_lgss_cython(model)
        res = flps_lgss(observations=np.array(model.obs, dtype=np.float64),
                        params=model.get_all_params(),
                        no_particles=self.settings['no_particles'],
                        fixed_lag=self.settings['fixed_lag'],
                        ancestors=self.results['ancestors'],
                        particles=self.results['self.particles'],
                        weights=self.results['self.weights'])
        self.results.update(res)
        self._estimate_gradient_and_hessian(model)
