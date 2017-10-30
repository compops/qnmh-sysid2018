"""Bootstrap particle filter for linear Gaussian model"""
import numpy as np

from state.particle_methods.resampling import multinomial
from state.particle_methods.resampling import stratified
from state.particle_methods.resampling import systematic

def bpf(particle_method, model):
    """Boostrap particle filter"""

    no_obs = model.no_obs + 1
    no_particles = particle_method.settings['no_particles']

    # Initalise variables
    ancestors = np.zeros((no_particles, no_obs))
    ancestors_resamp = np.zeros((no_particles, no_obs))
    particles = np.zeros((no_particles, no_obs))
    weights = np.zeros((no_particles, no_obs))
    filt_state_est = np.zeros((no_obs, 1))
    log_like = np.zeros(no_obs)

    # Generate or set initial state
    if particle_method.settings['generate_initial_state']:
        particles[:, 0] = model.generate_initial_state(no_particles)
        weights[:, 0] = 1.0 / no_particles
    else:
        particles[:, 0] = particle_method.settings['initial_state']
        weights[:, 0] = 1.0 / no_particles

    for i in range(1, no_obs):
        # Resample particles
        if particle_method.settings['resampling_method'] is 'multinomial':
            new_ancestors = multinomial(weights[:, i-1])
        elif particle_method.settings['resampling_method'] is 'stratified':
            new_ancestors = stratified(weights[:, i-1])
        elif particle_method.settings['resampling_method'] is 'systematic':
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
    return {'filt_state_est': filt_state_est,
            'log_like': np.sum(log_like),
            'particle_traj': particle_traj,
            'particles': particles,
            'weights': weights,
            'ancestors': ancestors,
            'ancestors_resampled': ancestors_resamp
           }