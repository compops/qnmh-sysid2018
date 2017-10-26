"""Cython helper for Kalman filtering."""

import numpy as np

def filter_helper(observations, params, initial_state, initial_cov):
    """Kalman filter."""
    no_obs = len(observations) - 1
    mu = params[0]
    phi = params[1]
    sigmav2 = params[2]
    sigmae2 = params[3]

    pred_state_est = np.zeros((no_obs + 1))
    pred_state_cov = np.zeros((no_obs + 1))
    filt_state_est = np.zeros((no_obs + 1))
    filt_state_cov = np.zeros((no_obs + 1))
    kalman_gain = np.zeros(no_obs + 1)
    log_like = 0.0

    filt_state_est[0] = initial_state
    filt_state_cov[0] = initial_cov

    for i in range(1, no_obs + 1):
        # Prediction step
        pred_state_est[i] = mu
        pred_state_est[i] += phi * (filt_state_est[i-1] - mu)
        pred_state_cov[i] = phi * filt_state_cov[i-1] * phi
        pred_state_cov[i] += sigmav2

        # Correction step
        pred_obs_cov = pred_state_cov[i] + sigmae2
        kalman_gain[i] = pred_state_cov[i] / pred_obs_cov

        scaled_innovation = (observations[i] - pred_state_est[i])
        scaled_innovation *= kalman_gain[i]
        filt_state_est[i] = pred_state_est[i] + scaled_innovation
        cov_change = kalman_gain[i] * pred_state_cov[i]
        filt_state_cov[i] = pred_state_cov[i] - cov_change

        mean = pred_state_est[i]
        stdev = np.sqrt(pred_obs_cov)
        log_like += norm_logpdf(observations[i], mean, stdev)

    return {'pred_state_est': pred_state_est,
            'pred_state_cov': pred_state_cov,
            'kalman_gain': kalman_gain,
            'filt_state_est': filt_state_est,
            'filt_state_cov': filt_state_cov,
            'log_like': log_like
            }

def norm_logpdf(parm, mean, stdev):
    """Helper for computing the log of the Gaussian pdf."""
    quad_term = 0.5 / (stdev**2) * (parm - mean)**2
    return -0.5 * np.log(2 * np.pi * stdev**2) - quad_term