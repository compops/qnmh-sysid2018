"""Cython helper for Kalman RTS smoothing."""

import numpy as np

def rts_helper(observations, params, pred_state_est, pred_state_cov,
               filt_state_est, filt_state_cov, kalman_gain,
               estimate_gradient=False, estimate_hessian=False):
    """Kalman RTS smoother."""
    no_obs = len(observations) - 1
    mu = params[0]
    phi = params[1]
    sigmav2 = params[2]
    sigmae2 = params[3]

    smo_gain = np.zeros((no_obs + 1, 1))
    smo_state_cov_twostep = np.zeros((no_obs + 1, 1))
    smo_state_est = np.zeros((no_obs + 1, 1))
    smo_state_cov = np.zeros((no_obs + 1, 1))
    gradient_part = []
    hessian = []

    # Run the preliminary Kalman filter
    smo_state_est[-1] = filt_state_est[-1]
    smo_state_cov[-1] = filt_state_cov[-1]

    for i in range((no_obs - 1), 0, -1):
        smo_gain[i] = filt_state_cov[i] * phi
        smo_gain[i] /= pred_state_cov[i+1]
        diff = smo_state_est[i+1] - pred_state_est[i+1]
        smo_state_est[i] = filt_state_est[i] + smo_gain[i] * diff
        diff = smo_state_cov[i+1] - pred_state_cov[i+1]
        smo_state_cov[i] = filt_state_cov[i]
        smo_state_cov[i] += smo_gain[i]**2 * diff

    if estimate_gradient or estimate_hessian:
        # Calculate the two-step smoothing covariance
        two_step = (1 - kalman_gain[-1]) * phi
        two_step *= filt_state_cov[-1]
        smo_state_cov_twostep[no_obs - 1] = two_step

        for i in range((no_obs - 1), 0, -1):
            term1 = filt_state_cov[i] * smo_gain[i-1]
            term2 = smo_gain[i-1]**2
            term3 = smo_state_cov_twostep[i+1]
            term4 = phi * filt_state_cov[i]
            smo_state_cov_twostep[i] = term1 + term2 * (term3 - term4)

    if estimate_gradient or estimate_hessian:
        # Gradient and Hessian estimation using Segal and Weinstein estimator
        gradient_part = np.zeros((4, no_obs))
        for i in range(1, no_obs):
            next_state = smo_state_est[i]
            cur_state = smo_state_est[i-1]
            eta = next_state * next_state + smo_state_cov[i]
            eta1 = cur_state**2 + smo_state_cov[i-1]
            psi = cur_state * next_state + smo_state_cov_twostep[i]
            quad_term = next_state - mu - phi * (cur_state - mu)
            isigmav2 = 1.0 / sigmav2

            gradient_part[0, i] = isigmav2 * quad_term * (1.0 - phi)

            term1 = isigmav2 * (1.0 - phi**2)
            term2 = psi - phi * eta1
            term2 -= cur_state * mu * (1.0 - 2.0 * phi)
            term2 += -next_state * mu + mu**2 * (1.0 - phi)
            gradient_part[1, i] = term1 * term2

            term1 = eta - 2 * phi * psi + phi**2 * eta1
            term2 = -2.0 * (next_state - phi * smo_state_est[i-1])
            term2 *= (1.0 - phi) * mu
            term3 = mu**2 * (1.0 - phi)**2
            gradient_part[2, i] = isigmav2 * (term1 + term2 + term3) - 1.0
            gradient_part[3, i] = 0.0

        gradient_sum = np.sum(gradient_part, axis=1)

    if estimate_hessian:
        part1 = np.mat(gradient_part).transpose()
        part1 = np.dot(np.mat(gradient_part), part1)
        part2 = np.mat(gradient_sum)
        part2 = np.dot(np.mat(gradient_sum).transpose(), part2)
        hessian = part1 - part2 / no_obs

    return {
            'smo_state_cov': smo_state_cov,
            'smo_state_est': smo_state_est,
            'gradient_part': gradient_part,
            'gradient_sum': gradient_sum,
            'hessian': hessian
           }