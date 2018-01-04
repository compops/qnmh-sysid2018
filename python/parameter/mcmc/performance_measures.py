###############################################################################
#    Constructing Metropolis-Hastings proposals using damped BFGS updates
#    Copyright (C) 2018  Johan Dahlin < uni (at) johandahlin [dot] com >
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
###############################################################################

"""Computes various performance meaures from runs of MCMC algorithms."""
import numpy as np

def compute_iact(mcmc, max_lag=None):
    """ Computes the integrated autocorrelation time (IACT).

        IACT is a standard metric to determine the mixing of an MCMC algorithm
        and is used to compare performance between such algorithms. It is
        computed as

            1 + 2 * sum_{k=1}^K \rho_k

        where \rho_k denotes the autocorrelation of the Markov chain at lag k.

        Args:
            mcmc: a Metropolis-Hastings object.
            max_lag: the maximum lag (K) to include in the IACT computation.

        Returns:
            An array with the IACT for each parameter in the Markov chain, i.e.,
            for each parameter to be estimated in the current model.

    """
    def helpter_iact(data, max_lag=None):
        """ Computes the ACF for a given data set. """
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
    """ Computes the efficient sample size (ESS).

        ESS is a standard metric to determine the mixing of an MCMC algorithm
        and is used to compare performance between such algorithms. It is
        computed as

            no_iters / (1 + 2 * sum_{k=1}^K \rho_k)

        where \rho_k denotes the autocorrelation of the Markov chain at lag k
        and no_iters denotes the number of iterations in the MH algorithm.

        Args:
            mcmc: a Metropolis-Hastings object.
            max_lag: the maximum lag (K) to include in the ESS computation.

        Returns:
            An array with the ESS for each parameter in the Markov chain, i.e.,
            for each parameter to be estimated in the current model.

    """
    burn_in_iters = mcmc.settings['no_burnin_iters']
    no_samples = mcmc.current_iter - burn_in_iters
    iact = compute_iact(mcmc, max_lag)
    return  no_samples / iact

def compute_sjd(mcmc):
    """ Computes the squared jump distance (SJD).

        SJD is a standard metric to determine the mixing of an MCMC algorithm
        and is used to compare performance between such algorithms. It is
        computed as

            1 / no_iters * sum_{k=2}^{no_iters} | \theta^(k) - \theta^(k-1) |

        where \theta^(k) denotes the state of the Markov chain at time k
        and no_iters denotes the number of iterations in the MH algorithm.

        Args:
            mcmc: a Metropolis-Hastings object.

        Returns:
            An array with the SJD for each parameter in the Markov chain, i.e.,
            for each parameter to be estimated in the current model.

    """
    burn_in_iters = mcmc.settings['no_burnin_iters']
    idx = range(int(burn_in_iters), int(mcmc.current_iter))
    trace = mcmc.free_params[idx, :]
    squared_jumps = np.linalg.norm(np.diff(trace, axis=0), 2, axis=1)**2

    return np.mean(squared_jumps)
