import numpy as np

# Zero-variance post processing with linear correction
def zvpost_linear_prototype(sampler):
    noParametersToEstimate = sampler.settings['noParametersToEstimate']
    noIters = sampler.settings['noIters']
    noBurnInIters = sampler.settings['noBurnInIters']
    
    ahat = np.zeros((noParametersToEstimate, noParametersToEstimate))
    
    for i in range(noParametersToEstimate):
        z = -0.5 * sampler.gradient[noBurnInIters:noIters, :]
        g = sampler.unrestrictedParameters[noBurnInIters:noIters, i]

        covAll = np.cov(np.vstack((z.transpose(), g.transpose())))
        Sigma = np.linalg.inv(covAll[0:noParametersToEstimate, 0:noParametersToEstimate])
        sigma = covAll[0:noParametersToEstimate, noParametersToEstimate]
        ahat[:, i] = - np.dot(Sigma, sigma)
    
    return sampler.unrestrictedParameters[noBurnInIters:noIters, :] + np.dot(z, ahat)