"""Computes various performance meaures from runs of MCMC algorithms."""
import numpy as np

def compute_iact(mcmc, max_lag=None):
    """Computes the integrated autocorrelation times for the parameters that
    are currently estimated by an MCMC algorithm."""

    def helpter_iact(data, max_lag=None):
        """Computes the ACF for a given data set."""
        no_data = len(data)
        variance = np.var(data)
        data = data - np.mean(data)
        correlations = np.correlate(data, data, mode='full')[-no_data:]
        result = correlations / (variance * (np.arange(no_data, 0, -1)))
        if not max_lag:
            max_lag = np.where(np.abs(result) < 1.96 / np.sqrt(no_data))
            if len(max_lag[0] > 0):
                max_lag = max_lag[0][0]
            else:
                max_lag = len(result)
        return 1.0 + 2.0 * np.sum(result[0:max_lag])

    output = np.zeros(mcmc.model.no_params_to_estimate)
    burn_in_iters = mcmc.settings['no_burnin_iters']
    idx = range(int(burn_in_iters), int(mcmc.current_iter))
    trace = mcmc.free_params[idx, :]
    for i in range(mcmc.model.no_params_to_estimate):
        output[i] = helpter_iact(trace[:, i], max_lag)
    return output

def compute_ess(mcmc, max_lag=None):
    """Computes the efficient sample sizes for the parameters that
    are currently estimated by an MCMC algorithm."""
    burn_in_iters = mcmc.settings['no_burnin_iters']
    no_samples = mcmc.current_iter - burn_in_iters
    iact = compute_iact(mcmc, max_lag)
    return  no_samples / iact

def compute_sjd(mcmc):
    """Computes the squared jump distances for the parameters that
    are currently estimated by an MCMC algorithm."""
    burn_in_iters = mcmc.settings['no_burnin_iters']
    idx = range(int(burn_in_iters), int(mcmc.current_iter))
    trace = mcmc.free_params[idx, :]
    squared_jumps = np.linalg.norm(np.diff(trace, axis=0), 2, axis=1)**2
    return np.mean(squared_jumps)
