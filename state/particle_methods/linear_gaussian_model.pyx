"""Bootstrap particle filter for linear Gaussian model"""
import numpy as np
from state.particle_methods.resampling import multinomial
from state.particle_methods.resampling import stratified
from state.particle_methods.resampling import systematic

def bpf_lgss(observations, params, no_particles, resampling_method,
             generate_initial_state=True, initial_state=None):
    """Boostrap particle filter for linear Gaussian model"""

    mu = params[0]
    phi = params[1]
    sigmav = params[2]
    sigmae = params[3]
    no_obs = len(observations)

    # Initalise variables
    ancestors = np.zeros((no_particles, no_obs))
    ancestors_resamp = np.zeros((no_particles, no_obs))
    particles = np.zeros((no_particles, no_obs))
    weights = np.zeros((no_particles, no_obs))
    filt_state_est = np.zeros((no_obs, 1))
    log_like = np.zeros(no_obs)

    # Generate or set initial state
    if generate_initial_state:
        noise_stdev = sigmav / np.sqrt(1.0 - phi**2)
        particles[:, 0] = mu + noise_stdev * np.random.normal(size=no_particles)
        weights[:, 0] = 1.0 / no_particles
    else:
        particles[:, 0] = initial_state
        weights[:, 0] = 1.0 / no_particles

    for i in range(1, no_obs):
        # Resample particles
        if resampling_method is 'multinomial':
            new_ancestors = multinomial(weights[:, i-1])
        elif resampling_method is 'stratified':
            new_ancestors = stratified(weights[:, i-1])
        elif resampling_method is 'systematic':
            new_ancestors = systematic(weights[:, i-1])
        else:
            raise ValueError("Unknown resampling method selected...")

        ancestors_resamp[:, 0:(i-1)] = ancestors_resamp[new_ancestors, 0:(i-1)]
        ancestors_resamp[:, i] = new_ancestors
        ancestors[:, i] = new_ancestors

        # Propagate particles
        mean = mu + phi * (particles[new_ancestors, i-1] - mu)
        particles[:, i] = mean + sigmav * np.random.normal(size=no_particles)

        # Weight particles
        unnormalised_weights = norm_logpdf(observations[i],
                                           mean=particles[:, i],
                                           stdev=sigmae)

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

def flps_lgss(observations, params, no_particles, fixed_lag, ancestors,
              particles, weights, estimate_gradient=False):
    """Fixed-lag particle smoother for linear Gaussian model"""

    mu = params[0]
    phi = params[1]
    sigmav = params[2]
    sigmae = params[3]
    no_obs = len(observations)

    # Initalise variables
    smo_state_est = np.zeros((no_obs, 1))
    smo_gradient_est = np.zeros((4, no_obs))
    log_joint_gradient_estimate = np.zeros(4)

    smo_state_est[0] = np.sum(particles[:, 0] * weights[:, 0])
    smo_state_est[-1] = np.sum(particles[:, -1] * weights[:, -1])

    # Run the fixed-lag smoother for the rest
    for i in range(0, no_obs-1):
        particle_indicies = np.arange(0, no_particles)
        lag = np.min((i + fixed_lag, no_obs - 1))

        # Reconstruct particle trajectory
        curr_ancestor = int(particle_indicies)
        for j in range(lag, i, -1):
            curr_ancestor = curr_ancestor.astype(int)
            next_ancestor = curr_ancestor
            curr_ancestor = ancestors[curr_ancestor, j].astype(int)

        # Estimate state
        weighted_particles = particles[curr_ancestor, i] * weights[:, lag]
        smo_state_est[i] = np.nansum(weighted_particles)

        # Estimate gradient
        if estimate_gradient:
            gradient_at_i = np.zeros((4, no_particles))
            state_quad_term = particles[next_ancestor, i+1]
            state_quad_term += -mu - phi * (particles[curr_ancestor, i] - mu)
            obs_quad_term = observations[i] - particles[curr_ancestor, i]
            q_matrix = sigmav**(-2)
            r_matrix = sigmae**(-2)

            gradient_at_i[0, :] = q_matrix * state_quad_term * (1.0 - phi)
            gradient_at_i[1, :] = q_matrix * state_quad_term
            gradient_at_i[1, :] *= (particles[curr_ancestor, i] - mu)
            gradient_at_i[1, :] *= (1.0 - phi**2)
            gradient_at_i[2, :] = q_matrix * state_quad_term**2 - 1.0
            gradient_at_i[3, :] = r_matrix * obs_quad_term**2 - 1.0

            for j in range(4):
                weighted_gradients = gradient_at_i[j, :] * weights[:, lag]
                smo_gradient_est[j, i] = np.nansum(weighted_gradients)

    if estimate_gradient:
        log_joint_gradient_estimate = np.sum(smo_gradient_est, axis=1)

    return {'smo_state_est': smo_state_est,
            'log_joint_gradient_estimate': log_joint_gradient_estimate
           }

def norm_logpdf(parm, mean, stdev):
    """Helper for computing the log of the Gaussian pdf."""
    quad_term = -0.5 / (stdev**2) * (parm - mean)**2
    return -0.5 * np.log(2 * np.pi * stdev**2) + quad_term
