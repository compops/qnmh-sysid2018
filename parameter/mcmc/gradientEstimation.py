import numpy as np

def getGradient(sampler, stateEstimator):
    if sampler.useGradientInformation:
        gradient = stateEstimator.gradientInternal
    else:
        gradient = np.zeros(sampler.settings['noParametersToEstimate'])
    return gradient

def getNaturalGradient(sampler, gradient, inverseHessian):
    flag = False
    if sampler.settings['memoryLength']:
        if sampler.currentIteration > sampler.settings['memoryLength']:
            flag = True
    else:
        flag = True
    
    if sampler.useGradientInformation and flag:
        stepSize = 0.5 * sampler.settings['noParametersToEstimate']**2
        naturalGradient = np.array(stepSize * np.dot(inverseHessian, gradient)).reshape(-1)
    else:
        naturalGradient = np.zeros(sampler.settings['noParametersToEstimate'])
    return naturalGradient