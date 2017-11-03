"""Helpers for computing Hessians for the use in the proposal distribution
of Metropolis-Hastings algorithms."""

import numpy as np
from helpers.cov_matrix import is_psd
from parameter.mcmc.quasi_newton.main import quasi_newton


def get_hessian(mcmc, state_estimator, prop_gradient=None):
    """ Computes the negative Hessian of the log-posterior.


        The negative Hessian of the log-posterior (observed information matrix)
        of the parameters to be estimated can either be a fixed matrix or be
        estimated using Kalman smoothing or Quasi-Newton methods depending
        on the settings of the MCMC method and the current iteration.

        If mcmc.settings['hessian_estimate'] is 'segal_weinstein', then the
        estimate is always obtained by the Kalman or particle smoother via
        the Segal-Weinstein estimator.

        If mcmc.settings['hessian_estimate'] is 'quasi_newton', one of two
        things can happen. The first thing is at the early stage of the MH
        algorithm when mcmc.current_iter > mcmc.settings['qn_memory_length'],
        then the estimate is determined by

            mcmc.settings['step_size']**2 * mcmc.settings['base_hessian']

        otherwise the estimate is computed using BFGS or SR1 updates and
        corrected using some method if the estimate is not positive semi-
        definite. See the documentation for quasi_newton and correct_hessian
        for more information about this.

        Args:
            mcmc: Metropolis-Hastings sampler object
            state_estimator: object for estimating the Hessian using Kalman or
                            particle smoothing methods.
            prop_gradient: the current proposed gradient
                           (to scale initial Hessian)

        Returns:
            An estimate of the negative inverse Hessian of the log-posterior.

    """
    step_size = mcmc.settings['step_size']**2

    # Default choice for Hessian from user
    inverse_hessian = step_size * mcmc.settings['base_hessian']

    # Estimate Hessian using Kalman smoothing or Quasi-Newton methods
    if mcmc.use_hess_info:
        if mcmc.settings['hessian_estimate'] is 'segal_weinstein':
            hessian_est = state_estimator.results['hessian_internal']
            inverse_hessian = np.linalg.inv(hessian_est)
            inverse_hessian *= step_size
            return correct_hessian(inverse_hessian, mcmc)
        if mcmc.settings['hessian_estimate'] is 'quasi_newton':
            if mcmc.current_iter > mcmc.settings['qn_memory_length']:
                inverse_hessian, no_samples = quasi_newton(mcmc, prop_gradient)
                if inverse_hessian is not None:
                    inverse_hessian *= step_size
                mcmc.no_samples_hess_est[mcmc.current_iter] = no_samples
                return correct_hessian(inverse_hessian, mcmc)

    if mcmc.settings['verbose']:
        print("Current inverse_hessian: " + str(inverse_hessian) + ".")
    return inverse_hessian


def correct_hessian(estimate, mcmc):
    """ Corrects estimate of the negative inverse Hessian of the log-posterior.

        The estimate is only corrected if it is not positive semi-definite. A
        number of different strategies can be used for this.

        if mcmc.settings['hessian_correction'] is 'replace': then the estimate
        is replaced by

        mcmc.settings['step_size']**2 * mcmc.settings['base_hessian']

        during the burn-in phase of the MH algorithm. Otherwise an estimate
        of the posterior covariance is computed during the latter part of
        the burn-in phase. This estimate is known as the empricial estimate
        of the posterior covariance and it replaces an incorrect estimate
        of the negative inverse Hessian.

        if mcmc.settings['hessian_correction'] is 'regularise': then the
        estimate is regularised by adding a diagonal matrix where the elements
        for two times the negative value of the smallest eigenvalue. As the
        estimate is only corrected when at least one eigenvalue is negative,
        this corresponds by adding a positive value to the diagonal elements
        which shifts the eigenvalues.

        if mcmc.settings['hessian_correction'] is 'flip': then the an
        eigenvalue-eigenvector decomposition is made. The diagonal matrix
        of the eigenvalues is changed to its absolute value (hence flipping
        the eigenvalues to the positive side) and the corrected estimate is
        computed by multiplying the decomposition back together using the
        adjusted diagonal matrix of the eigenvalues.

        Args:
            estimate: an estimate of the negative inverse Hessian of the
                      log-posterior.
            mcmc: Metropolis-Hastings sampler object

        Returns:
            A corrected estimate (positive definite) of the negative inverse
            Hessian of the log-posterior.

    """
    strategy = mcmc.settings['hessian_correction']

    # No correction
    if not strategy:
        return estimate

    if estimate is None or not is_psd(estimate):
        mcmc.no_hessians_corrected += 1
        mcmc.iter_hessians_corrected.append(mcmc.current_iter)

        # if is_psd(-estimate):
        #     print("Iteration: " + str(mcmc.current_iter) +
        #           ", switched to negative Hessian estimate...")
        #     return -estimate

        if strategy is 'replace' or estimate is None:
            if mcmc.current_iter > mcmc.settings['no_burnin_iters']:
                if mcmc.settings['hessian_correction_verbose']:
                    print("Iteration: " + str(mcmc.current_iter) +
                          ", corrected Hessian by replacing with estimate " +
                          "from latter half of burn-in.")

                if not hasattr(mcmc, 'emp_hessian'):
                    idx = range(int(0.5 * mcmc.settings['no_burnin_iters']),
                                mcmc.settings['no_burnin_iters'])
                    trace = mcmc.free_params[idx, :]
                    mcmc.emp_hessian = np.cov(trace, rowvar=False)
                    print("Iteration: " + str(mcmc.current_iter) +
                          ", computed an empirical estimate of the posterior "
                          + "covariance to replace ND Hessian estimates.")
                return mcmc.emp_hessian
            else:
                step_size = mcmc.settings['step_size']**2
                return step_size * mcmc.settings['base_hessian']

        # Add a diagonal matrix proportional to the largest negative eigenvalue
        elif strategy is 'regularise':
            min_eigval = np.min(np.linalg.eig(estimate)[0])
            if mcmc.settings['hessian_correction_verbose']:
                print("Iteration: " + str(mcmc.current_iter) +
                      ", corrected Hessian by adding diagonal matrix " +
                      "with elements: " + str(-2.0 * min_eigval))
            corrected_estimate = estimate
            corrected_estimate -= 2.0 * min_eigval * np.eye(estimate.shape[0])
            return corrected_estimate

        # Flip the negative eigenvalues
        elif strategy is 'flip':
            if mcmc.settings['hessian_correction_verbose']:
                print("Iteration: " + str(mcmc.current_iter) +
                      ", corrected Hessian by flipping negative eigenvalues " +
                      "to positive.")
            evd = np.linalg.eig(estimate)
            ev_matrix = np.diag(np.abs(evd[0]))
            estimate = np.matmul(evd[1], ev_matrix)
            estimate = np.matmul(estimate, evd[1])
            return estimate
        else:
            raise ValueError("Unknown Hessian correction strategy...")
    else:
        return estimate
