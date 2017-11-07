from __future__ import absolute_import

import cython

from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log, sqrt, exp, isfinite
from libc.float cimport FLT_MAX
from libc.stdlib cimport malloc, free

DEF FixedLag = 10
DEF NoParticles = 1000
DEF NoObs = 501
DEF PI = 3.1415

@cython.cdivision(True)
@cython.boundscheck(False)
def bpf_lgss(double [:] obs, double mu, double phi, double sigmav, double sigmae):

    # Initialise variables
    cdef int[NoParticles] ancestors
    cdef double[NoParticles] new_part
    cdef double[NoParticles] old_part
    cdef double[NoParticles] weights
    cdef double[NoObs] filt_state_est
    cdef double[NoParticles] unnorm_weights
    cdef double[NoParticles] shifted_weights
    cdef double log_like = 0.0

    # Define helpers
    cdef double mean = 0.0
    cdef double stDev = 0.0
    cdef double max_weight = 0.0
    cdef double norm_factor
    cdef double foo_double

    # Define counters
    cdef int i
    cdef int j

    # Generate or set initial state
    stDev = sigmav / sqrt(1.0 - (phi * phi))
    for j in range(NoParticles):
        old_part[j] = mu + stDev * random_gaussian()
        weights[j] = 1.0 / NoParticles

    for i in range(1, NoObs):

        # Resample particles
        systematic(ancestors, weights)

        # Propagate particles
        for j in range(NoParticles):
            mean = mu + phi * (old_part[ancestors[j]] - mu)
            new_part[j] = mean + sigmav * random_gaussian()

        # Weight particles
        for j in range(NoParticles):
            unnorm_weights[j] = norm_logpdf(obs[i], new_part[j], sigmae)

        max_weight = my_max(unnorm_weights)
        norm_factor = 0.0
        for j in range(NoParticles):
            shifted_weights[j] = exp(unnorm_weights[j] - max_weight)
            foo_double = norm_factor + shifted_weights[j]
            if isfinite(foo_double) != 0:
                norm_factor = foo_double

        # Normalise weights and compute state filtering estimate
        filt_state_est[i] = 0.0
        for j in range(NoParticles):
            weights[j] = shifted_weights[j] / norm_factor
            if isfinite(weights[j] * new_part[j]) != 0:
                filt_state_est[i] += weights[j] * new_part[j]

        # Estimate log-likelihood
        log_like += max_weight + log(norm_factor) - log(NoParticles)

        # Set new to old
        for j in range(NoParticles):
            old_part[j] = new_part[j]

    # Compile the rest of the output
    return filt_state_est, log_like

@cython.cdivision(True)
@cython.boundscheck(False)
def flps_lgss(double [:] obs, double mu, double phi, double sigmav, double sigmae):

    # Initialise variables
    cdef int *ancestors = <int *>malloc(NoParticles * sizeof(int))
    cdef double *particle_history = <double *>malloc(FixedLag * NoParticles * sizeof(double))
    cdef double *old_particle_history = <double *>malloc(FixedLag * NoParticles * sizeof(double))

    cdef double *particles = <double *>malloc(NoObs * NoParticles * sizeof(double))
    cdef double *weights = <double *>malloc(NoObs * NoParticles * sizeof(double))
    cdef double *weights_at_t = <double *>malloc(NoParticles * sizeof(double))

    cdef double[NoObs] filt_state_est
    cdef double[NoObs] smo_state_est

    cdef double *unnorm_weights = <double *>malloc(NoParticles * sizeof(double))
    cdef double *shifted_weights = <double *>malloc(NoParticles * sizeof(double))

    cdef double sub_gradient[4]
    cdef double gradient[4][NoObs]

    cdef double log_like = 0.0

    # Define helpers
    cdef double mean = 0.0
    cdef double stDev = 0.0
    cdef double max_weight = 0.0
    cdef double norm_factor
    cdef double foo_double

    cdef double q_matrix = 1.0 / (sigmav * sigmav)
    cdef double r_matrix = 1.0 / (sigmae * sigmae)
    cdef double state_quad_term = 0.0
    cdef double curr_particle = 0.0
    cdef double next_particle = 0.0

    # Define counters
    cdef int i
    cdef int j
    cdef int current_lag
    cdef int k
    cdef int idx
    cdef int idx_curr
    cdef int idx_next
    cdef int idx_t

    # Initialize ancestry
    for k in range(FixedLag):
        for j in range(NoParticles):
            particle_history[k + j * FixedLag] = 0.0
            old_particle_history[k + j * FixedLag] = 0.0

    for i in range(NoObs):
        filt_state_est[i] = 0.0
        smo_state_est[i] = 0.0
        for j in range(NoParticles):
            weights_at_t[j] = 0.0
            particles[i + j * NoObs] = 0.0
            weights[i + j * NoObs] = 0.0

    for i in range(4):
        sub_gradient[i] = 0.0
        for j in range(NoObs):
            gradient[i][j] = 0.0

    # Generate initial state
    stDev = sigmav / sqrt(1.0 - (phi * phi))
    for j in range(NoParticles):
        particles[0 + j * NoObs] = mu + stDev * random_gaussian()
        weights[0 + j * NoObs] = 1.0 / NoParticles
        particle_history[0 + j * FixedLag] = particles[0 + j * NoObs]

        filt_state_est[0] = 0.0
        for j in range(NoParticles):
            filt_state_est[0] += weights[0 + j * NoObs] * particles[0 + j * NoObs]

    for i in range(1, NoObs):
        current_lag = my_min(i, FixedLag)

        # Resample particles
        for j in range(NoParticles):
            weights_at_t[j] = weights[i - 1 + j * NoObs]
        systematic(ancestors, weights_at_t)

        # Update ancestry
        for j in range(NoParticles):
            for k in range(FixedLag):
                old_particle_history[k + j * FixedLag] = particle_history[k + j * FixedLag]

        # Propagate particles
        for j in range(NoParticles):
            mean = mu + phi * (particles[i - 1 + ancestors[j] * NoObs] - mu)
            particles[i + j * NoObs] = mean + sigmav * random_gaussian()
            particle_history[0 + j * FixedLag] = particles[i + j * NoObs]
            for k in range(1, FixedLag):
                particle_history[k + j * FixedLag] = old_particle_history[k - 1 + ancestors[j] * FixedLag]

        # Weight particles
        for j in range(NoParticles):
            unnorm_weights[j] = norm_logpdf(obs[i], particles[i + j * NoObs], sigmae)

        max_weight = my_max(unnorm_weights)
        norm_factor = 0.0
        for j in range(NoParticles):
            shifted_weights[j] = exp(unnorm_weights[j] - max_weight)
            foo_double = norm_factor + shifted_weights[j]
            if isfinite(foo_double) != 0:
                norm_factor = foo_double

        # Normalise weights and compute state filtering estimate
        for j in range(NoParticles):
            weights[i + j * NoObs] = shifted_weights[j] / norm_factor
            if isfinite(weights[i + j * NoObs] * particles[i + j * NoObs]) != 0:
                filt_state_est[i] += weights[i + j * NoObs] * particles[i + j * NoObs]

        # Compute smoothed state
        if i >= FixedLag:
            for j in range(NoParticles):
                curr_particle = particle_history[(FixedLag - 1) + j * FixedLag]
                next_particle = particle_history[(FixedLag - 2) + j * FixedLag]

                smo_state_est[i - FixedLag + 1] += weights[i + j * NoObs] * curr_particle

                state_quad_term = next_particle - mu - phi * (curr_particle - mu)
                sub_gradient[0] = q_matrix * state_quad_term * (1.0 - phi)
                sub_gradient[1] = q_matrix * state_quad_term * (curr_particle - mu) * (1.0 - phi**2)
                sub_gradient[2] = q_matrix * state_quad_term * state_quad_term - 1.0
                sub_gradient[3] = 0.0

                gradient[0][i - FixedLag + 1] += sub_gradient[0] * weights[i + j * NoObs]
                gradient[1][i - FixedLag + 1] += sub_gradient[1] * weights[i + j * NoObs]
                gradient[2][i - FixedLag + 1] += sub_gradient[2] * weights[i + j * NoObs]
                gradient[3][i - FixedLag + 1] += sub_gradient[3] * weights[i + j * NoObs]

        # Estimate log-likelihood
        log_like += max_weight + log(norm_factor) - log(NoParticles)

    # Estimate gradients of the log joint distribution
    for i in range(NoObs - FixedLag, NoObs):
        idx  = NoObs - i - 1
        for j in range(NoParticles):
            curr_particle = particle_history[idx + j * FixedLag]
            smo_state_est[i] +=  weights[NoObs - 1 + j * NoObs] * curr_particle

            if (idx - 1) >= 0:
                next_particle = particle_history[idx - 1 + j * FixedLag]
                state_quad_term = next_particle - mu - phi * (curr_particle - mu)
                sub_gradient[0] = q_matrix * state_quad_term * (1.0 - phi)
                sub_gradient[1] = q_matrix * state_quad_term * (curr_particle - mu) * (1.0 - phi**2)
                sub_gradient[2] = q_matrix * state_quad_term * state_quad_term - 1.0
                sub_gradient[3] = 0.0

                gradient[0][i - FixedLag + 1] += sub_gradient[0] * weights[i + j * NoObs]
                gradient[1][i - FixedLag + 1] += sub_gradient[1] * weights[i + j * NoObs]
                gradient[2][i - FixedLag + 1] += sub_gradient[2] * weights[i + j * NoObs]
                gradient[3][i - FixedLag + 1] += sub_gradient[3] * weights[i + j * NoObs]

    free(particles)
    free(weights)
    free(weights_at_t)
    free(particle_history)
    free(old_particle_history)
    free(ancestors)
    free(unnorm_weights)
    free(shifted_weights)

    # Compile the rest of the output
    return filt_state_est, smo_state_est, log_like, gradient

@cython.cdivision(True)
@cython.boundscheck(False)
cdef double norm_logpdf(double x, double m, double s):
    """Helper for computing the log of the Gaussian pdf."""
    cdef double part1 = -0.91893853320467267 # -0.5 * log(2 * pi)
    cdef double part2 = -log(s)
    cdef double part3 = -0.5 * (x - m) * (x - m) / (s * s)
    return part1 + part2 + part3

@cython.cdivision(True)
@cython.boundscheck(False)
cdef void systematic(int *ancestors, double weights[NoParticles]):
    cdef int cur_idx = 0
    cdef int j = 0
    cdef double rnd_number = random_uniform()
    cdef double cpoint = 0.0
    cdef double[NoParticles] cum_weights
    cdef double sum_weights

    # Compute the empirical CDF of the weights
    cum_weights[0] = weights[0]
    sum_weights = weights[0]
    for j in range(1, NoParticles):
        cum_weights[j] = cum_weights[j-1] + weights[j]
        sum_weights += weights[j]

    for j in range(1, NoParticles):
        cum_weights[j] /= sum_weights

    for j in range(NoParticles):
        cpoint = (rnd_number + j) / NoParticles
        while cum_weights[cur_idx] < cpoint and cur_idx < NoParticles - 1:
            cur_idx += 1
        ancestors[j] = cur_idx

@cython.cdivision(True)
@cython.boundscheck(False)
cdef double random_uniform():
    cdef double r = rand()
    return r / RAND_MAX

@cython.cdivision(True)
@cython.boundscheck(False)
cdef double random_gaussian():
    cdef double x1, x2, w

    w = 2.0
    while (w >= 1.0):
        x1 = 2.0 * random_uniform() - 1.0
        x2 = 2.0 * random_uniform() - 1.0
        w = x1 * x1 + x2 * x2

    w = sqrt((-2.0 * log(w)) / w)
    return x1 * w

@cython.boundscheck(False)
cdef double my_max(double weights[NoParticles]):
    cdef int idx = 0
    cdef int i = 0
    cdef double current_largest = weights[0]

    for i in range(1, NoParticles):
        if weights[i] > current_largest and isfinite(weights[i]):
            idx = i
    return weights[idx]

@cython.boundscheck(False)
cdef int my_min(int x, int y):
    cdef int foo
    if x > y or x == y:
        foo = x
    else:
        foo = y
    return foo

