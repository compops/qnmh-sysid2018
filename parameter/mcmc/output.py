import numpy as np
import os
import sys

from scipy.linalg import eigh
from scipy.stats._multivariate import _eigvalsh_to_eps
from scipy.special import gammaln

import matplotlib.pylab as plt
from palettable.colorbrewer.qualitative import Dark2_8

# Print small progress reports
def print_progress_report(mcmc, max_iact_lag=100):
    iteration = mcmc.current_iter

    print("################################################################################################ ")
    print(" Iteration: " + str(iteration + 1) + " of : " + str(mcmc.settings['no_iters']) + " completed.")
    print("")
    print(" Current state of the Markov chain:               ")
    print(["%.4f" % v for v in mcmc.params[iteration - 1, :]])
    print("")
    print(" Proposed next state of the Markov chain:         ")
    print(["%.4f" % v for v in mcmc.prop_params[iteration, :]])
    print("")
    print(" Current posterior mean estimate: ")
    print(["%.4f" % v for v in np.mean(mcmc.params[range(iteration), :], axis=0)])
    print("")
    print(" Current acceptance rate:                         ")
    print("%.4f" % np.mean(mcmc.accepted[range(iteration)]))
    if (iteration > (mcmc.settings['no_burnin_iters'] * 1.5)):
        print("")
        print(" Current IACT values:                         ")
        print(["%.2f" % v for v in mcmc.compute_iact()])
        print("")
        print(" Current log-SJD value:                          ")
        print(str(np.log(mcmc.compute_sjd())))
    if mcmc.settings['hessian_estimate'] is not 'kalman':
        if (iteration > mcmc.settings['qn_memory_length']):
            no_samples_hess_est = mcmc.no_samples_hess_est[range(iteration)]
            idx = np.where(no_samples_hess_est > 0)[0]
            if len(idx) > 0:
                print("")
                print(" Mean number of samples for Hessian estimate:           ")
                print("%.4f" % np.mean(no_samples_hess_est[idx]))

    print("################################################################################################ ")

def plot_results(mcmc):
    noBins = int(np.floor(np.sqrt(len(mcmc.results['trace'][:, 0]))))
    noParameters = mcmc.model.no_params_to_estimate
    parameterNames = mcmc.model.params_to_estimate

    plt.figure()
    for i in range(noParameters):
        plt.subplot(noParameters, 4, 4 * i + 1)
        plt.hist(mcmc.results['trace'][:, i], bins=noBins, color = Dark2_8.mpl_colors[i])
        plt.ylabel("Marginal posterior probability of " + parameterNames[i])
        plt.xlabel("iteration")
        plt.subplot(noParameters, 4, 4 * i + 2)
        plt.plot(mcmc.results['trace'][:, i], color = Dark2_8.mpl_colors[i])
        plt.ylabel("Parameter trace of " + parameterNames[i])
        plt.xlabel("iteration")
        plt.subplot(noParameters, 4, 4 * i + 3)
        plt.plot(mcmc.results['prop_params'][:, i], color = Dark2_8.mpl_colors[i])
        plt.ylabel("Proposed trace of " + parameterNames[i])
        plt.xlabel("iteration")
        plt.subplot(noParameters, 4, 4 * i + 4)
        plt.plot(mcmc.results['prop_grad'][:, i], color = Dark2_8.mpl_colors[i])
        plt.ylabel("natural gradient of " + parameterNames[i])
        plt.xlabel("iteration")
    plt.show()