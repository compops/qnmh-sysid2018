"""Particle methods."""
import numpy as np
from state.particle_methods.cython_helper import bpf_lgss, flps_lgss
from state.base_state_inference import BaseStateInference

class ParticleMethodsCythonLGSS(BaseStateInference):
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
        obs = np.array(model.obs.flatten())
        params = model.get_all_params()
        state, ll = bpf_lgss(obs, mu=params[0], phi=params[1], sigmav=params[2], sigmae=params[3])
        self.results.update({'filt_state_est': np.array(state).reshape((model.no_obs+1, 1)), 'log_like': ll})

    def smoother(self, model):
        """Fixed-lag particle smoother for linear Gaussian model."""
        self.name = "Fixed-lag particle smoother (Cython)"
        obs = np.array(model.obs.flatten())
        params = model.get_all_params()
        xhatf, xhats, ll, gradient = flps_lgss(obs, mu=params[0],
                                               phi=params[1], sigmav=params[2],
                                               sigmae=params[3])

        # Compute estimate of gradient and Hessian
        gradient = np.array(gradient).reshape((4, model.no_obs+1))
        log_joint_gradient_estimate = np.sum(gradient, axis=1)

        part1 = np.mat(gradient).transpose()
        part1 = np.dot(np.mat(gradient), part1)
        part2 = np.mat(log_joint_gradient_estimate)
        part2 = np.dot(np.mat(log_joint_gradient_estimate).transpose(), part2)
        log_joint_hessian_estimate = part1 - part2 / model.no_obs

        self.results.update({'filt_state_est': np.array(xhatf).reshape((model.no_obs+1, 1))})
        self.results.update({'log_like': ll})
        self.results.update({'smo_state_est': np.array(xhats).reshape((model.no_obs+1, 1))})
        self.results.update({'log_joint_gradient_estimate': log_joint_gradient_estimate})
        self.results.update({'log_joint_hessian_estimate': log_joint_hessian_estimate})

        self._estimate_gradient_and_hessian(model)
