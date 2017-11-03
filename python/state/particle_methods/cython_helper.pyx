"""Bootstrap particle filter for linear Gaussian model"""
from __future__ import absolute_import

import numpy as np
cimport numpy as np

DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t

DTYPE_float = np.float64
ctypedef np.float64_t DTYPE_float_t

from state.particle_methods.resampling import systematic

def bpf_lgss(np.ndarray[DTYPE_float_t, ndim=2] observations,
             np.ndarray[DTYPE_float_t] params,
             int no_particles):
    """Boostrap particle filter for linear Gaussian model"""

    cdef float mu = params[0]
    cdef float phi = params[1]
    cdef float sigmav = params[2]
    cdef float sigmae = params[3]
    cdef int no_obs = len(observations)

    # Initalise variables
    cdef np.ndarray[DTYPE_int_t, ndim=2] ancestors = np.zeros((no_particles, no_obs), dtype=DTYPE_int)
    cdef np.ndarray[DTYPE_int_t, ndim=2] ancestors_resamp = np.zeros((no_particles, no_obs), dtype=DTYPE_int)
    cdef np.ndarray[DTYPE_float_t, ndim=2] particles = np.zeros((no_particles, no_obs))
    cdef np.ndarray[DTYPE_float_t, ndim=2] weights = np.zeros((no_particles, no_obs))
    cdef np.ndarray[DTYPE_float_t, ndim=2] filt_state_est = np.zeros((no_obs, 1))
    cdef np.ndarray[DTYPE_float_t] log_like = np.zeros(no_obs)
    cdef np.ndarray[DTYPE_float_t] renorm_weight

    cdef np.ndarray[DTYPE_float_t] mean
    cdef np.ndarray[DTYPE_float_t] unnormalised_weights
    cdef float max_weight
    cdef float normalisation_factor
    cdef np.ndarray[DTYPE_float_t] shifted_weights

    cdef int particle_index
    cdef np.ndarray[DTYPE_float_t, ndim=2] particle_traj

    # Generate or set initial state
    noise_stdev = sigmav / np.sqrt(1.0 - phi**2)
    particles[:, 0] = mu + noise_stdev * np.random.normal(size=no_particles)
    weights[:, 0] = 1.0 / no_particles

    for i in range(1, no_obs):
        # Resample particles
        new_ancestors = systematic(weights[:, i-1]).astype(int)
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
    renorm_weight = weights[:, no_obs-1] / np.sum(weights[:, no_obs-1])
    particle_index = np.random.choice(no_particles, 1, p=renorm_weight)
    particle_traj = particles[ancestors_resamp[particle_index, :], :]

    # Compile the rest of the output
    return {'filt_state_est': filt_state_est,
            'log_like': np.sum(log_like),
            'particle_traj': particle_traj,
            'particles': particles,
            'weights': weights,
            'ancestors': ancestors,
            'ancestors_resampled': ancestors_resamp
           }


def flps_lgss(np.ndarray[DTYPE_float_t, ndim=2] observations,
              np.ndarray[DTYPE_float_t] params,
              int no_particles,
              int fixed_lag,
              np.ndarray[DTYPE_float_t, ndim=2] ancestors,
              np.ndarray[DTYPE_float_t, ndim=2] particles,
              np.ndarray[DTYPE_float_t, ndim=2] weights):
    """Fixed-lag particle smoother for linear Gaussian model"""

    cdef float mu = params[0]
    cdef float phi = params[1]
    cdef float sigmav = params[2]
    cdef float sigmae = params[3]
    cdef int no_obs = len(observations)

    # Initalise variables
    cdef np.ndarray[DTYPE_float_t, ndim=2] smo_state_est = np.zeros((no_obs, 1))
    cdef np.ndarray[DTYPE_float_t, ndim=2] smo_gradient_est = np.zeros((4, no_obs))
    cdef np.ndarray[DTYPE_float_t] log_joint_gradient_estimate = np.zeros(4)
    cdef np.ndarray[DTYPE_float_t, ndim=2] gradient_at_i = np.zeros((4, no_particles))

    smo_state_est[0] = np.sum(particles[:, 0] * weights[:, 0])
    smo_state_est[no_obs-1] = np.sum(particles[:, no_obs-1] * weights[:, no_obs-1])

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
        state_quad_term = particles[next_ancestor, i+1] - mu
        state_quad_term -= phi * (particles[curr_ancestor, i] - mu)
        q_matrix = sigmav**(-2)
        r_matrix = sigmae**(-2)

        gradient_at_i[0, :] = q_matrix * state_quad_term * (1.0 - phi)
        gradient_at_i[1, :] = q_matrix * state_quad_term
        gradient_at_i[1, :] *= (particles[curr_ancestor, i] - mu)
        gradient_at_i[1, :] *= (1.0 - phi**2)
        gradient_at_i[2, :] = q_matrix * state_quad_term**2 - 1.0
        gradient_at_i[3, :] = 0.0

        for j in range(4):
            weighted_gradients = gradient_at_i[j, :] * weights[:, lag]
            smo_gradient_est[j, i] = np.nansum(weighted_gradients)

    log_joint_gradient_estimate = np.sum(smo_gradient_est, axis=1)

    return {'smo_state_est': smo_state_est,
            'log_joint_gradient_estimate': log_joint_gradient_estimate
           }

def norm_logpdf(float parm, np.ndarray[DTYPE_float_t] mean, float stdev):
    """Helper for computing the log of the Gaussian pdf."""
    cdef np.ndarray[DTYPE_float_t] quad_term = -0.5 / (stdev**2) * (parm - mean)**2
    return -0.5 * np.log(2 * np.pi * stdev**2) + quad_term
