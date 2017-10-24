import numpy as np
import os
import sys

from scipy.linalg import eigh
from scipy.stats._multivariate import _eigvalsh_to_eps
from scipy.special import gammaln

import matplotlib.pylab as plt
from palettable.colorbrewer.qualitative import Dark2_8


# Print small progress reports
def print_progress_report(sampler, maxIACTLag=100):
    iteration = sampler.currentIteration

    print("################################################################################################ ")
    print(" Iteration: " + str(iteration + 1) + " of : " + str(sampler.settings['noIters']) + " completed.")
    print("")
    print(" Current state of the Markov chain:               ")
    print(["%.4f" % v for v in sampler.restrictedParameters[iteration - 1, :]])
    print("")
    print(" Proposed next state of the Markov chain:         ")
    print(["%.4f" % v for v in sampler.proposedRestrictedParameters[iteration, :]])
    print("")
    print(" Current posterior mean estimate: ")
    print(["%.4f" % v for v in np.mean(sampler.restrictedParameters[range(iteration), :], axis=0)])
    print("")
    print(" Current acceptance rate:                         ")
    print("%.4f" % np.mean(sampler.accepted[range(iteration)]))
    if (iteration > (sampler.settings['noBurnInIters'] * 1.5)):
        print("")
        print(" Current IACT values:                         ")
        print(["%.2f" % v for v in sampler.calcIACT()])
        print("")
        print(" Current log-SJD value:                          ")
        print(str(np.log(sampler.calcSJD())))
    if sampler.settings['hessianEstimate'] is not 'kalman':
        if (iteration > sampler.settings['memoryLength']):
            noEffectiveSamples = sampler.noEffectiveSamples[range(iteration)]
            idx = np.where(noEffectiveSamples > 0)[0]
            if len(idx) > 0:
                print("")
                print(" Mean number of samples for Hessian estimate:           ")
                print("%.4f" % np.mean(noEffectiveSamples[idx]))

    print("################################################################################################ ")





def plot_results(sampler):
    noBins = int(np.floor(np.sqrt(len(sampler.results['parameterTrace'][:, 0]))))
    noParameters = sampler.settings['noParametersToEstimate']
    parameterNames = sampler.settings['parametersToEstimate']

    plt.figure()
    for i in range(noParameters):
        plt.subplot(noParameters, 4, 4 * i + 1)
        plt.hist(sampler.results['parameterTrace'][:, i], bins=noBins, color = Dark2_8.mpl_colors[i])
        plt.ylabel("Marginal posterior probability of " + parameterNames[i])
        plt.xlabel("iteration")
        plt.subplot(noParameters, 4, 4 * i + 2)
        plt.plot(sampler.results['parameterTrace'][:, i], color = Dark2_8.mpl_colors[i])
        plt.ylabel("Parameter trace of " + parameterNames[i])
        plt.xlabel("iteration")
        plt.subplot(noParameters, 4, 4 * i + 3)
        plt.plot(sampler.results['proposedRestrictedParameters'][:, i], color = Dark2_8.mpl_colors[i])
        plt.ylabel("Proposed trace of " + parameterNames[i])
        plt.xlabel("iteration")
        plt.subplot(noParameters, 4, 4 * i + 4)
        plt.plot(sampler.results['proposedNaturalGradient'][:, i], color = Dark2_8.mpl_colors[i])
        plt.ylabel("natural gradient of " + parameterNames[i])
        plt.xlabel("iteration")
    plt.show()

def truncateContribution(x, limit):
    if not limit:
        return x

    if isinstance(x, float):
        sign = np.sign(x)
        output = sign * np.min((limit, np.abs(x)))

    if isinstance(x, np.ndarray):
        output = np.zeros(len(x))
        for i in range(len(x)):
            sign = np.sign(x[i])
            output[i] = sign * np.min((limit, np.abs(x[i])))

    return output

