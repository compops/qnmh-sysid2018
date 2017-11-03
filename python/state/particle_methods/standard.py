"""Particle methods."""
import numpy as np
from state.particle_methods.resampling import multinomial
from state.particle_methods.resampling import stratified
from state.particle_methods.resampling import systematic
from state.base_state_inference import BaseStateInference

class ParticleMethods(BaseStateInference):
    """Particle methods."""

    def __init__(self, new_settings=None):
        self.name = "Particle methods"
        self.settings = {'no_particles': 100,
                         'resampling_method': 'systematic',
                         'fixed_lag': 0,
                         'initial_state': 0.0,
                         'generate_initial_state': False,
                         'estimate_gradient': False,
                         'verbose': False
                         }
        if new_settings:
            self.settings.update(new_settings)

    def filter(self, model):
        """Bootstrap particle filter"""
        self.name = "Bootstrap particle filter"
        no_obs = model.no_obs + 1
        no_particles = self.settings['no_particles']

        if self.settings['verbose']:
            print("")
            print("Particle filter running with model parameters:")
            print(["%.3f" % v for v in model.get_all_params()])

        # Initalise variables
        ancestors = np.zeros((no_particles, no_obs))
        ancestors_resamp = np.zeros((no_particles, no_obs))
        particles = np.zeros((no_particles, no_obs))
        weights = np.zeros((no_particles, no_obs))
        filt_state_est = np.zeros((no_obs, 1))
        log_like = np.zeros(no_obs)

        # Generate or set initial state
        if self.settings['generate_initial_state']:
            particles[:, 0] = model.generate_initial_state(no_particles)
            weights[:, 0] = 1.0 / no_particles
        else:
            particles[:, 0] = self.settings['initial_state']
            weights[:, 0] = 1.0 / no_particles

        for i in range(1, no_obs):
            # Resample particles
            if self.settings['resampling_method'] is 'multinomial':
                new_ancestors = multinomial(weights[:, i-1])
            elif self.settings['resampling_method'] is 'stratified':
                new_ancestors = stratified(weights[:, i-1])
            elif self.settings['resampling_method'] is 'systematic':
                new_ancestors = systematic(weights[:, i-1])
            else:
                raise ValueError("Unknown resampling method selected...")

            ancestors_resamp[:, 0:(i-1)] = ancestors_resamp[new_ancestors, 0:(i-1)]
            ancestors_resamp[:, i] = new_ancestors
            ancestors[:, i] = new_ancestors

            # Propagate particles
            particles[:, i] = model.generate_state(particles[new_ancestors, i-1], i)

            # Weight particles
            unnormalised_weights = model.evaluate_obs(particles[:, i], i)

            max_weight = np.max(unnormalised_weights)
            shifted_weights = np.exp(unnormalised_weights - max_weight)
            normalisation_factor = np.sum(shifted_weights)
            weights[:, i] = shifted_weights / normalisation_factor

            # Estimate log-likelihood
            log_like[i] = max_weight
            log_like[i] += np.log(normalisation_factor)
            log_like[i] -= np.log(no_particles)

            # Estimate the filtered state
            filt_state_est[i] = np.sum(weights[:, i] * particles[:, i])

        # Sample a trajectory
        particle_index = np.random.choice(no_particles, 1, p=weights[:, -1])
        ancestory_trajectory = ancestors_resamp[particle_index, -1].astype(int)
        particle_traj = particles[ancestory_trajectory, :]

        # Compile the rest of the output
        self.results.update({'filt_state_est': filt_state_est,
                             'log_like': np.sum(log_like),
                             'particle_traj': particle_traj
                            })
        self.particles = particles
        self.weights = weights
        self.ancestors = ancestors
        self.ancestors_resamp = ancestors_resamp

        if self.settings['verbose']:
            print("Log-likelihood estimate is: " + str(self.results['log_like']))

    def smoother(self, model):
        """Fixed-lag particle smoother"""
        self.name = "Bootstrap particle filter and fixed-lag particle smoother."
        no_obs = model.no_obs + 1
        no_params = model.no_params
        no_particles = self.settings['no_particles']
        fixed_lag = self.settings['fixed_lag']

        self.filter(model)
        particles = self.particles
        weights = self.weights
        ancestors = self.ancestors

        # Initalise variables
        smo_state_est = np.zeros((no_obs, 1))
        smo_gradient_est = np.zeros((no_params, no_obs))
        log_joint_gradient_estimate = np.zeros(no_params)
        log_joint_hessian_estimate = np.zeros((no_params, no_params))

        smo_state_est[0] = self.results['filt_state_est'][0]
        smo_state_est[-1] = self.results['filt_state_est'][-1]

        # Run the fixed-lag smoother for the rest
        for i in range(0, no_obs-1):
            particle_indicies = np.arange(0, no_particles)
            lag = int(np.min((i + fixed_lag, no_obs - 1)))

            # Reconstruct particle trajectory
            curr_ancestor = particle_indicies
            for j in range(lag, i, -1):
                curr_ancestor = curr_ancestor.astype(int)
                next_ancestor = curr_ancestor.astype(int)
                curr_ancestor = ancestors[curr_ancestor, j].astype(int)

            # Estimate state
            weighted_particles = particles[curr_ancestor, i] * weights[:, lag]
            smo_state_est[i] = np.nansum(weighted_particles)

            # Estimate gradient
            if self.settings['estimate_gradient']:
                sub_grad = model.log_joint_gradient(particles[next_ancestor, i+1],
                                                    particles[curr_ancestor, i],
                                                    i)

                j = 0
                for param in sub_grad:
                    weighted_gradients = sub_grad[param] * weights[:, lag]
                    smo_gradient_est[j, i] = np.nansum(weighted_gradients)
                    j += 1

        if self.settings['estimate_gradient']:
            log_joint_gradient_estimate = np.sum(smo_gradient_est, axis=1)

            part1 = np.mat(smo_gradient_est).transpose()
            part1 = np.dot(np.mat(smo_gradient_est), part1)
            part2 = np.matmul(smo_gradient_est, smo_gradient_est.transpose())
            log_joint_hessian_estimate = part1 - part2 / no_obs

        self.results.update({'smo_state_est': smo_state_est,
                             'log_joint_gradient_estimate': log_joint_gradient_estimate,
                             'log_joint_hessian_estimate': log_joint_hessian_estimate
                            })

        if self.settings['estimate_gradient']:
            self._estimate_gradient_and_hessian(model)

