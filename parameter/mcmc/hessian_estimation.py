"""Helpers for computing Hessians for the use in the proposal distribution
of Metropolis-Hastings algorithms."""

import numpy as np
from helpers.cov_matrix import is_psd

def get_hessian(mho, state_estimator, prop_gradient=None):
    """Get the Hessian of the log-posterior of the parameters as a fixed
    matrix or estimate it using Kalman smoothing or Quasi-Newton methods.

    Args:
        mho: Metropolis-Hastings sampler object
        state_estimator: object for estimating the Hessian using Kalman or
                         particle smoothing methods.
        prop_gradient: the current proposed gradient (to scale initial Hessian)

    Returns:
        An estimate of the negative inverse Hessian of the log-posterior.
    """
    step_size = mho.settings['stepSize']**2

    # Default choice for Hessian from user
    inverse_hessian = np.eye(mho.settings['no_params_to_estimate'])
    inverse_hessian *= mho.settings['initialHessian']**2
    inverse_hessian *= step_size

    # Estimate Hessian using Kalman smoothing or Quasi-Newton methods
    if mho.use_hessian_information:
        if mho.settings['hessian_estimate'] is 'kalman':
            inverse_hessian = np.linalg.inv(state_estimator.hessian_internal)
            inverse_hessian *= step_size
            return _correct_hessian(inverse_hessian, mho)
        if mho.settings['hessian_estimate'] is 'quasiNewton':
            if mho.current_iter > mho.settings['mem_length']:
                inverse_hessian = _quasi_newton(mho, prop_gradient)
                inverse_hessian *= step_size
                return _correct_hessian(inverse_hessian, mho)

    if mho.settings['verbose']:
        print("Current inverse_hessian: " + str(inverse_hessian) + ".")
    return inverse_hessian

def _correct_hessian(estimate, mho):
    """Corrects a Hessian estimate that is not postive definite.

    Args:
        estimate: an estimate of the negative inverse Hessian of the
                  log-posterior.
        mho: Metropolis-Hastings sampler object

    Returns:
        A corrected estimate (positive definite) of the negative inverse Hessian
        of the log-posterior.
    """
    strategy = mho.settings['hessianCorrectionstrategy']

    # No correction
    if not strategy:
        return estimate

    if isinstance(estimate, bool) or not is_psd(estimate):
        mho.no_hessians_corrected += 1
        mho.iter_hessians_corrected.append(mho.current_iter)

        if is_psd(-estimate):
            print("Iteration: " + str(mho.current_iter) +
                  ", switched to negative Hessian estimate...")
            return -estimate

        if strategy is 'replace':
            if mho.current_iter > mho.settings['no_burnin_iters']:
                if mho.settings['hessian_correction_verbose']:
                    print("Iteration: " + str(mho.current_iter) +
                          ", corrected Hessian by replacing with estimate from " +
                          "latter half of burn-in.")

                if not hasattr(mho, 'emp_hessian'):
                    idx = range(int(0.5 * mho.settings['no_burnin_iters']),
                                mho.settings['no_burnin_iters'])
                    trace = mho.free_params[idx, :]
                    mho.emp_hessian = np.cov(trace, rowvar=False)
                    print("Iteration: " + str(mho.current_iter) +
                          ", computed an empirical estimate of the posterior "
                          + "covariance to replace ND Hessian estimates.")
                return mho.emp_hessian
            else:
                identity_matrix = np.diag(np.ones(mho.no_params_to_estimate))
                return identity_matrix * mho.settings['initialHessian']**2

        # Add a diagonal matrix proportional to the largest negative eigenvalue
        elif strategy is 'regularise':
            min_eigval = np.min(np.linalg.eig(estimate)[0])
            if mho.settings['hessian_correction_verbose']:
                print("Iteration: " + str(mho.current_iter) +
                      ", corrected Hessian by adding diagonal matrix " +
                      " with elements: " + str(-2.0 * min_eigval))
            return estimate - 2.0 * min_eigval * np.eye(estimate.shape[0])

        # Flip the negative eigenvalues
        elif strategy is 'flip':
            if mho.settings['hessian_correction_verbose']:
                print("Iteration: " + str(mho.current_iter) +
                      ", corrected Hessian by flipping negative eigenvalues " +
                      "to positive.")
            evd = np.linalg.eig(estimate)
            ev_matrix = np.diag(np.abs(evd[0]))
            return np.dot(np.dot(evd[1], ev_matrix), evd[1])
        else:
            raise ValueError("Unknown Hessian correction strategy...")
    else:
        return estimate

def _quasi_newton(mho, prop_gradient):
    """Implements Quasi-Newton methods for Hessian estimation."""
    mem_length = mho.settings['quasi_newton']['memory_length']
    base_hessian = mho.settings['base_hessian']
    strategy = mho.settings['quasi_newton']['strategy']
    only_accepted_info = mho.settings['quasi_newton']['only_accepted_info']
    no_params = mho.no_params_to_estimtae

    # Extract parameters and gradients
    idx = range(mho.current_iter - mem_length, mho.current_iter)
    parameters = mho.prop_free_params[idx, :]
    gradients = mho.prop_gradient[idx, :]
    hessians = mho.prop_hessian[idx, :, :]
    accepted = mho.accepted[idx]
    target = mho.prop_log_prior[idx] + mho.prop_log_likelihood[idx]
    target = np.concatenate(target).reshape(-1)

    # Keep only unique parameters and gradients
    if only_accepted_info:
        idx = np.where(accepted > 0)[0]

        # No available infomation, so quit
        if len(idx) is 0:
            if mho.settings['verbose']:
                print("Not enough samples to estimate Hessian...")
            if mho.settings['hessian_correction'] is 'replace':
                return _correct_hessian(True, mho)
            else:
                return base_hessian

        parameters = parameters[idx, :]
        gradients = gradients[idx, :]
        hessians = hessians[idx, :, :]
        target = target[idx]
        accepted = accepted[idx, :]

    # Sort and compute differences
    idx = np.argsort(target)
    parameters = parameters[idx, :]
    gradients = gradients[idx, :]
    hessians = np.matmul(hessians[idx, :], hessians[idx, :])

    param_diff = np.zeros((len(idx) - 1, no_params))
    grad_diff = np.zeros((len(idx) - 1, no_params))

    for i in range(len(idx) - 1):
        param_diff[i, :] = parameters[i+1, :] - parameters[i, :]
        grad_diff[i, :] = gradients[i+1, :] - gradients[i, :]

    initial_hessian = _init_hessian_estimate(mho, prop_gradient,
                                             param_diff, grad_diff)

    if strategy is 'bfgs':
        estimate, no_samples = _bfgs_estimate(initial_hessian, mho,
                                              param_diff, grad_diff)
        return _correct_hessian(estimate, mho), no_samples
    elif strategy is 'sr1':
        estimate, no_samples = _sr1_estimate(initial_hessian, param_diff,
                                             grad_diff)
        return _correct_hessian(estimate, mho), no_samples

    else:
        raise NameError("Unknown quasi-Newton algorithm selected...")

def _bfgs_estimate(estimate, mho, param_diff, grad_diff):
    """Implements BFGS update for Hessian estimation."""
    curv_cond = mho.settings['quasiNewton']['bfgs_curvature_cond']
    no_params = mho.no_params_to_estimtae
    identity_matrix = np.diag(np.ones(no_params))
    no_samples = 0
    violate_curv_cond = 0

    for i in range(param_diff.shape[0]):
        do_update = False

        if curv_cond is 'enforce':
            if np.dot(param_diff[i], grad_diff[i]) < 0.0:
                do_update = True
                new_grad_diff = grad_diff[i]
            else:
                violate_curv_cond += 1

        elif curv_cond is 'damped':
            term1 = np.dot(param_diff[i], grad_diff[i])
            term2 = np.dot(param_diff[i], np.linalg.inv(estimate))
            term2 = np.dot(term2, param_diff[i])
            if term1 > 0.2 * term2:
                theta = 1.0
            else:
                theta = 0.8 * term2 / (term2 - term1)
            grad_guess = np.dot(np.linalg.inv(estimate), param_diff[i])
            new_grad_diff = theta * grad_diff[i] + (1.0 - theta) * grad_guess
            do_update = True

        elif curv_cond is 'ignore':
            do_update = True
            new_grad_diff = grad_diff[i]
        else:
            raise NameError("Unknown flag curv_cond given to function")

        if do_update:
            no_samples += 1
            rho = 1.0 / np.dot(new_grad_diff, param_diff[i])
            term1 = np.outer(param_diff[i], new_grad_diff)
            term1 = identity_matrix - rho * term1
            term2 = np.outer(new_grad_diff, param_diff[i])
            term2 = identity_matrix - rho * term2
            term3 = rho * np.outer(param_diff[i], param_diff[i])

            tmp_term1 = np.matmul(term1, estimate)
            estimate = np.matmul(tmp_term1, term2) + term3

    #print("BFGS, noMaxSamples: " + str(len(param_diff)) + ", no_samples: "
    #       + str(no_samples) + " and violate_curv_cond: "
    #       + str(violate_curv_cond) + ".")
    return -estimate, no_samples

def _sr1_estimate(estimate, param_diff, grad_diff):
    """Implements SR1 update for Hessian estimation."""
    no_samples = 0

    for i in range(param_diff.shape[0]):
        diff_term = param_diff[i] - np.dot(estimate, grad_diff[i])
        if np.dot(diff_term, grad_diff[i]) != 0.0:
            rank1_update = np.outer(diff_term, diff_term)
            rank1_update /= np.dot(diff_term, grad_diff[i])
            estimate += rank1_update
            no_samples += 1

    return -estimate, no_samples

def _init_hessian_estimate(mho, prop_gradient, param_diff, grad_diff):
    """Implements different strategies to initialise the Hessian."""
    strategy = mho.settings['quasi_newton']['initial_hessian']
    scaling = mho.settings['quasi_newton']['initial_hessian_scaling']
    fixed_hessian = mho.settings['quasi_newton']['initial_hessian_fixed']
    no_params = mho.no_params_to_estimate
    identity_matrix = np.diag(np.ones(no_params))

    if strategy is 'fixed':
        return fixed_hessian * identity_matrix

    if strategy is 'scaled_gradient':
        return identity_matrix * scaling / np.linalg.norm(prop_gradient, 2)

    if strategy is 'scaled_curvature':
        scaled_curvature = np.dot(param_diff[0], grad_diff[0])
        scaled_curvature *= np.dot(grad_diff[0], grad_diff[0])
        return identity_matrix * np.abs(scaled_curvature)
