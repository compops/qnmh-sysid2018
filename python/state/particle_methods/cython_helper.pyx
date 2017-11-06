from __future__ import absolute_import

import cython

from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log, sqrt, exp, isfinite
from libc.float cimport FLT_MAX
from libc.stdlib cimport malloc, free

DEF FixedLag = 10
DEF NoParticles = 500
DEF NoObs = 1001
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
    cdef int[NoParticles] ancestors
    cdef int[FixedLag][NoParticles] ancestry
    cdef int[FixedLag][NoParticles] old_ancestry
    cdef double *particles = <double *>malloc(NoObs * NoParticles * sizeof(double))
    #cdef double[NoObs][NoParticles] particles
    cdef double[NoObs][NoParticles] weights
    cdef double[NoObs] filt_state_est
    cdef double[NoObs] smo_state_est
    cdef double[NoParticles] unnorm_weights
    cdef double[NoParticles] shifted_weights
    cdef double[4][NoParticles] gradient_at_i
    cdef double[4] log_joint_gradient_estimate
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
    cdef int current_lag
    cdef int k

    # Initialize ancestry
    for k in range(FixedLag):
        for j in range(NoParticles):
            ancestry[k][j] = 0

    # Generate initial state
    stDev = sigmav / sqrt(1.0 - (phi * phi))
    for j in range(NoParticles):
        particles[0 + j * NoObs] = mu + stDev * random_gaussian()
        weights[0][j] = 1.0 / NoParticles
        ancestry[0][j] = j

        filt_state_est[0] = 0.0
        for j in range(NoParticles):
            filt_state_est[0] += weights[0][j] * particles[0 + j * NoObs]

    for i in range(1, NoObs):
        current_lag = my_min(i, FixedLag)

        # Resample particles
        systematic(ancestors, weights[i-1])

        # Update ancestry
        for k in range(FixedLag):
            for j in range(NoParticles):
                old_ancestry[k][j] = ancestry[k][j]

        for j in range(NoParticles):
            idx = ancestors[j]
            ancestry[0][j] = idx
            for k in range(1, FixedLag):
                ancestry[k][j] = old_ancestry[k-1][idx]

        # Propagate particles
        for j in range(NoParticles):
            mean = mu + phi * (particles[i - 1 + ancestors[j] * NoObs] - mu)
            particles[i + j * NoObs] = mean + sigmav * random_gaussian()

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
        filt_state_est[i] = 0.0
        for j in range(NoParticles):
            weights[i][j] = shifted_weights[j] / norm_factor
            if isfinite(weights[i][j] * particles[i + j * NoObs]) != 0:
                filt_state_est[i] += weights[i][j] * particles[i + j * NoObs]

        # Compute smoothed state
        if i >= FixedLag:
            filt_state_est[i-FixedLag] = 0.0
            for j in range(NoParticles):
                idx = ancestry[FixedLag-1][j]
                smo_state_est[i-FixedLag] += weights[i][j] * particles[i - FixedLag + idx * NoObs]

        # Estimate log-likelihood
        log_like += max_weight + log(norm_factor) - log(NoParticles)

    for i in range(NoObs-FixedLag, NoObs):
        smo_state_est[i] = 0.0
        for j in range(NoParticles):
            idx = ancestry[NoObs-i-1][j]
            smo_state_est[i] += weights[NoObs-1][j] * particles[i + idx * NoObs]

    free(particles)

    # Compile the rest of the output
    return filt_state_est, smo_state_est, log_like

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
