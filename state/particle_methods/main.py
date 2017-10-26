"""Particle methods."""
import numpy as np
from scipy.stats import norm
from state.particle_methods.linear_gaussian_model import bpf_lgss, flps_lgss

class ParticleMethods(object):
    """Particle methods."""

    def __init__(self, new_settings=None):
        self.name = None
        self.settings = {}
        self.settings.update({'generate_initial_state': True})
        self.settings.update({'initial_state': 0.0})
        self.settings.update({'resampling_method': 'systematic'})
        self.settings.update({'no_particles': 1000})
        self.settings.update({'estimate_gradient': False})
        self.settings.update({'fixed_lag': 10})

        self.log_like = []
        self.filt_state_est = []
        self.smo_state_est = []
        self.xtraj = []

        self.gradient = []
        self.gradient_internal = []

        if new_settings:
            self.settings.update(new_settings)

    def filter_linear_gaussian_model(self, model):
        """Bootstrap particle filter for linear Gaussian model."""
        self.name = "Bootstrap particle filter"
        output = bpf_lgss(observations=model.obs,
                          params=model.get_all_params(),
                          no_particles=self.settings['no_particles'],
                          resampling_method=self.settings['resampling_method'],
                          generate_initial_state=self.settings['generate_initial_state'],
                          initial_state=self.settings['initial_state']
                         )

        self.filt_state_est = output['filt_state_est']
        self.log_like = output['log_like']
        self.x_traj = output['particle_traj']
        self.particles = output['particles']
        self.weights = output['weights']
        self.ancestors = output['ancestors']
        self.ancestors_resampled = output['ancestors_resampled']

    def smoother_linear_gaussian_model(self, model):
        """Fixed-lag particle smoother for linear Gaussian model."""
        self.name = "Fixed-lag particle smoother"
        self.filter_linear_gaussian_model(model)

        output = flps_lgss(observations=model.obs,
                           params=model.get_all_params(),
                           no_particles=self.settings['no_particles'],
                           fixed_lag=self.settings['fixed_lag'],
                           ancestors=self.ancestors,
                           particles=self.particles,
                           weights=self.weights,
                           estimate_gradient=self.settings['estimate_gradient']
                          )

        self.smo_state_est = output['smo_state_est']
        gradient_estimate = output['log_joint_gradient_estimate']

        gradient = {}
        gradient_internal = []
        i = 0
        for parameter in model.params.keys():
            if parameter in model.params_to_estimate:
                gradient.update({parameter: gradient_estimate[i]})
                gradient_internal.append(gradient_estimate[i])
            i += 1

        # Add the log-prior derivatives
        if self.settings['estimate_gradient']:
            log_prior_gradient = model.log_prior_gradient()
            i = 0
            for first_param in log_prior_gradient.keys():
                gradient_estimate[i] += log_prior_gradient[first_param]
                i += 1

        if self.settings['estimate_gradient']:
            self.gradient_internal = np.array(gradient_internal)
            self.gradient = gradient