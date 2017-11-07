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
        log_joint_hessian_estimate = part1 - part2 / model.no_obs

        self.results.update({'pred_state_est': np.array(xhatp).reshape((model.no_obs+1, 1))})
        self.results.update({'pred_state_cov': np.array(Pp).reshape((model.no_obs+1, 1))})
        self.results.update({'filt_state_est': np.array(xhatf).reshape((model.no_obs+1, 1))})
        self.results.update({'filt_state_cov': np.array(Pf).reshape((model.no_obs+1, 1))})
        self.results.update({'smo_state_est': np.array(xhats).reshape((model.no_obs+1, 1))})
        self.results.update({'smo_state_cov': np.array(Ps).reshape((model.no_obs+1, 1))})
        self.results.update({'log_like': float(ll)})
        self.results.update({'log_joint_gradient_estimate': log_joint_gradient_estimate})
        self.results.update({'log_joint_hessian_estimate': log_joint_hessian_estimate})

        self._estimate_gradient_and_hessian(model)
