from __future__ import absolute_import

import cython

from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log, sqrt, exp, isfinite
from libc.float cimport FLT_MAX
from libc.stdlib cimport malloc, free

DEF NoObs = 1001
DEF PI = 3.1415

@cython.cdivision(True)
@cython.boundscheck(False)
def kf_filter(double [:] obs, double mu, double phi, double sigmav, double sigmae,
              double initial_state, double initial_cov):

    cdef double[NoObs] pred_state_est
    cdef double[NoObs] pred_state_cov
    cdef double[NoObs] filt_state_est
    cdef double[NoObs] filt_state_cov
    cdef double[NoObs] kalman_gain
    cdef double log_like = 0.0
    cdef int i = 0

    cdef double scaled_innovation = 0.0
    cdef double cov_change = 0.0
    cdef double pred_obs_cov = 0.0

    for i in range(NoObs):
        pred_state_est[i] = 0.0
        pred_state_cov[i] = 0.0
        filt_state_est[i] = 0.0
        kalman_gain[i] = 0.0

    filt_state_est[0] = initial_state
    filt_state_cov[0] = initial_cov

    for i in range(1, NoObs):
        # Prediction step
        pred_state_est[i] = mu
        pred_state_est[i] += phi * (filt_state_est[i-1] - mu)
        pred_state_cov[i] = phi * filt_state_cov[i-1] * phi
        pred_state_cov[i] += sigmav * sigmav

        # Correction step
        pred_obs_cov = pred_state_cov[i] + sigmae * sigmae
        kalman_gain[i] = pred_state_cov[i] / pred_obs_cov

        scaled_innovation = (obs[i] - pred_state_est[i])
        scaled_innovation *= kalman_gain[i]
        filt_state_est[i] = pred_state_est[i] + scaled_innovation
        cov_change = kalman_gain[i] * pred_state_cov[i]
        filt_state_cov[i] = pred_state_cov[i] - cov_change

        log_like += norm_logpdf(obs[i], pred_state_est[i], sqrt(pred_obs_cov))

    return pred_state_est, pred_state_cov, filt_state_est, filt_state_cov, log_like


@cython.cdivision(True)
@cython.boundscheck(False)
def rts_smoother(double [:] obs, double mu, double phi, double sigmav, double sigmae,
                 double initial_state, double initial_cov):

    cdef double[NoObs] pred_state_est
    cdef double[NoObs] pred_state_cov
    cdef double[NoObs] filt_state_est
    cdef double[NoObs] filt_state_cov
    cdef double[NoObs] smo_state_est
    cdef double[NoObs] smo_state_cov_twostep
    cdef double[NoObs] smo_state_cov
    cdef double[NoObs] smo_gain
    cdef double[NoObs] kalman_gain
    cdef double[4][NoObs] gradient_part
    cdef double log_like = 0.0
    cdef int i

    cdef double scaled_innovation = 0.0
    cdef double cov_change = 0.0
    cdef double pred_obs_cov = 0.0
    cdef double diff = 0.0
    cdef double term1 = 0.0
    cdef double term2 = 0.0
    cdef double term3 = 0.0
    cdef double term4 = 0.0
    cdef double two_step = 0.0

    cdef double next_state = 0.0
    cdef double cur_state = 0.0
    cdef double eta = 0.0
    cdef double eta1 = 0.0
    cdef double psi = 0.0
    cdef double quad_term = 0.0
    cdef double isigmav2 = 0.0

    filt_state_est[0] = initial_state
    filt_state_cov[0] = initial_cov

    for i in range(NoObs):
        pred_state_est[i] = 0.0
        pred_state_cov[i] = 0.0
        filt_state_est[i] = 0.0
        filt_state_cov[i] = 0.0
        smo_state_est[i] = 0.0
        smo_state_cov_twostep[i] = 0.0
        smo_state_cov[i] = 0.0
        smo_gain[i] = 0.0
        kalman_gain[i] = 0.0

    for i in range(4):
        for j in range(NoObs):
            gradient_part[i][j] = 0.0

    # Filter
    for i in range(1, NoObs):
        # Prediction step
        pred_state_est[i] = mu
        pred_state_est[i] += phi * (filt_state_est[i-1] - mu)
        pred_state_cov[i] = phi * filt_state_cov[i-1] * phi
        pred_state_cov[i] += sigmav * sigmav

        # Correction step
        pred_obs_cov = pred_state_cov[i] + sigmae * sigmae
        kalman_gain[i] = pred_state_cov[i] / pred_obs_cov

        scaled_innovation = (obs[i] - pred_state_est[i])
        scaled_innovation *= kalman_gain[i]
        filt_state_est[i] = pred_state_est[i] + scaled_innovation
        cov_change = kalman_gain[i] * pred_state_cov[i]
        filt_state_cov[i] = pred_state_cov[i] - cov_change

        log_like += norm_logpdf(obs[i], pred_state_est[i], sqrt(pred_obs_cov))

    # Smoother
    smo_state_est[NoObs - 1] = filt_state_est[NoObs - 1]
    smo_state_cov[NoObs - 1] = filt_state_cov[NoObs - 1]

    for i in range((NoObs - 2), 0, -1):
        smo_gain[i] = filt_state_cov[i] * phi / pred_state_cov[i+1]
        smo_state_est[i] = filt_state_est[i] + smo_gain[i] * (smo_state_est[i+1] - pred_state_est[i+1])
        smo_state_cov[i] = filt_state_cov[i] + smo_gain[i]**2 * (smo_state_cov[i+1] - pred_state_cov[i+1])

    # Calculate the two-step smoothing covariance
    two_step = (1.0 - kalman_gain[NoObs - 1]) * phi * filt_state_cov[NoObs - 1]
    smo_state_cov_twostep[NoObs - 1] = two_step

    for i in range((NoObs - 1), 0, -1):
        term1 = filt_state_cov[i] * smo_gain[i-1]
        term2 = smo_gain[i-1] * smo_gain[i-1]
        term3 = smo_state_cov_twostep[i+1]
        term4 = phi * filt_state_cov[i]
        smo_state_cov_twostep[i] = term1 + term2 * (term3 - term4)

    # Gradient and Hessian estimation using Segal and Weinstein estimator
    for i in range(1, NoObs):
        next_state = smo_state_est[i]
        cur_state = smo_state_est[i-1]
        eta = next_state * next_state + smo_state_cov[i]
        eta1 = cur_state * cur_state + smo_state_cov[i-1]
        psi = cur_state * next_state + smo_state_cov_twostep[i]
        quad_term = next_state - mu - phi * (cur_state - mu)
        isigmav2 = 1.0 / (sigmav * sigmav)

        gradient_part[0][i] = isigmav2 * (1.0 - phi) * quad_term

        term1 = isigmav2 * (1.0 - phi * phi)
        term2 = psi - phi * eta1
        term2 -= cur_state * mu * (1.0 - 2.0 * phi)
        term2 += - next_state * mu + mu * mu * (1.0 - phi)
        gradient_part[1][i] = term1 * term2

        term1 = eta - 2 * phi * psi + phi * phi * eta1
        term2 = -2.0 * (next_state - phi * smo_state_est[i-1])
        term2 *= (1.0 - phi) * mu
        term3 = mu * mu * (1.0 - phi) * (1.0 - phi)
        gradient_part[2][i] = isigmav2 * (term1 + term2 + term3) - 1.0
        gradient_part[3][i] = 0.0

    return pred_state_est, pred_state_cov, filt_state_est, filt_state_cov, log_like, smo_state_est, smo_state_cov, gradient_part

@cython.cdivision(True)
@cython.boundscheck(False)
cdef double norm_logpdf(double x, double m, double s):
    """Helper for computing the log of the Gaussian pdf."""
    cdef double part1 = -0.91893853320467267 # -0.5 * log(2 * pi)
    cdef double part2 = -log(s)
    cdef double part3 = -0.5 * (x - m) * (x - m) / (s * s)
    return part1 + part2 + part3