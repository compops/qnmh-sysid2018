import numpy as np
from parameter.mcmc.helpers import isPositiveSemiDefinite

def getHessian(sampler, stateEstimator):
    inverseHessian = np.eye(sampler.settings['noParametersToEstimate']) 
    inverseHessian *= sampler.settings['initialHessian']

    if sampler.useHesssianInformation:
        if sampler.settings['hessianEstimate'] is 'kalman':
            return np.linalg.inv(correctHessian(stateEstimator.hessianInternal))
        if sampler.currentIteration > sampler.settings['memoryLength']:
            if sampler.settings['hessianEstimate'] is 'BFGS' or sampler.settings['hessianEstimate'] is 'SR1':            
                return estimateHessianQN(sampler, sampler.settings['hessianEstimate'])
    
    if sampler.settings['verbose']:
        print("Current inverseHessian: " + str(inverseHessian) + ".")    
    return inverseHessian

def correctHessian(x, approach='regularise'):

    if not isPositiveSemiDefinite(x):
        # Add a diagonal matrix proportional to the largest negative eigenvalue
        if approach is 'regularise':
            smallestEigenValue = np.min(np.linalg.eig(x)[0])
            x -= 2.0 * smallestEigenValue * np.eye(x.shape[0])
            #print("Corrected Hessian by adding diagonal matrix with elements: " + str(-2.0 * smallestEigenValue))

        # Flip the negative eigenvalues
        if approach is 'flip':
            evDecomp = np.linalg.eig(x)
            x = np.dot(np.dot(evDecomp[1], np.diag(np.abs(evDecomp[0]))), evDecomp[1])
            #print("Corrected Hessian by flipping negative eigenvalues to positive.")
    
    return(x)


def estimateHessianQN(sampler, method='BFGS', useInformationFromRejectedSteps=False):
    memoryLength = sampler.settings['memoryLength']
    initialHessian = sampler.settings['initialHessian']
    noParameters = sampler.noParametersToEstimate
    identityMatrix = np.diag(np.ones(noParameters))
   
    # Extract parameters and gradients
    idx = range(sampler.currentIteration - memoryLength, sampler.currentIteration)
    parameters = sampler.proposedParameters[idx, :]
    gradients = sampler.proposedGradient[idx, :]
    hessians = sampler.proposedHessian[idx, :, :]
    accepted = sampler.accepted[idx]
    target = np.concatenate(sampler.proposedLogPrior[idx] + sampler.proposedLogLikelihood[idx]).reshape(-1)

    # Keep only unique parameters and gradients
    if not useInformationFromRejectedSteps:
        idx = np.where(accepted > 0)[0]

        # No available infomation, so quit
        if len(idx) == 0:
            return identityMatrix * initialHessian**2
        
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
    
    parametersDiff = np.zeros((len(idx) - 1, noParameters))
    gradientsDiff = np.zeros((len(idx) - 1, noParameters))

    for i in range(len(idx) - 1):
        parametersDiff[i, :] = parameters[i+1, :] - parameters[i, :]
        gradientsDiff[i, :] = gradients[i+1, :] - gradients[i, :]

    if method is 'DampedBFGS':
        inverseHessianEstimate, noEffectiveSamples = estimateHessianBFGS(sampler, parametersDiff, gradientsDiff, dampedBFGS=True)

    if method is 'BFGS':
        inverseHessianEstimate, noEffectiveSamples = estimateHessianBFGS(sampler, parametersDiff, gradientsDiff)
    
    if method is 'SR1':
        inverseHessianEstimate, noEffectiveSamples = estimateHessianBFGS(sampler, parametersDiff, gradientsDiff)

    sampler.noEffectiveSamples[sampler.currentIteration] = noEffectiveSamples
    return inverseHessianEstimate

def estimateHessianBFGS(sampler, parametersDiff, gradientsDiff, dampedBFGS=False):
    memoryLength = sampler.settings['memoryLength']
    initialHessian = sampler.settings['initialHessian']
    noParameters = sampler.noParametersToEstimate
    identityMatrix = np.diag(np.ones(noParameters))

    # Initialisation of H0
    if initialHessian is 'scaledProposedGradient':
        proposedGradient = sampler.proposedGradient[sampler.currentIteration, :]
        inverseHessianEstimate = identityMatrix * initialHessian / np.linalg.norm(proposedGradient, 2)

    if initialHessian is 'scaledCurvature':
        scaledCurvature = np.dot(parametersDiff[0], gradientsDiff[0]) * np.dot(gradientsDiff[0], gradientsDiff[0])
        inverseHessianEstimate = identityMatrix * np.abs(scaledCurvature)

    if isinstance(initialHessian, float):
        inverseHessianEstimate = initialHessian**2 * identityMatrix
    
    noEffectiveSamples = 0

    for i in range(parametersDiff.shape[0]):
        doUpdate = False

        if dampedBFGS:
            term1 = np.dot(parametersDiff[i], gradientsDiff[i])
            term2 = np.dot(np.dot(parametersDiff[i], inverseHessianEstimate), parametersDiff[i])

            if (term1 > 0.2 * term2):
                theta = 1.0
            else:
                theta = 0.8 * term2 / (term2 - term1)
            
            r = theta * gradientsDiff[i] + (1.0 - theta) * np.dot(inverseHessianEstimate, parametersDiff[i])
            doUpdate = True
        else:
            if np.dot(parametersDiff[i], gradientsDiff[i]) > 0:
                doUpdate = True
                r = gradientsDiff[i]

        if doUpdate:
            quadraticFormSB = np.dot(np.dot(parametersDiff[i], inverseHessianEstimate), parametersDiff[i])
            curvatureCondition = np.dot(parametersDiff[i], r)

            term1 = (curvatureCondition + quadraticFormSB) / curvatureCondition**2
            term1 *= np.outer(parametersDiff[i], parametersDiff[i])

            term2 = np.dot(np.dot(inverseHessianEstimate, r), parametersDiff[i])
            term2 += np.dot(np.dot(parametersDiff[i], r), inverseHessianEstimate)
            term2 /= curvatureCondition

            inverseHessianEstimate += term1 - term2
            noEffectiveSamples += 1
    
    return inverseHessianEstimate, noEffectiveSamples


def estimateHessianSR1(sampler, parametersDiff, gradientsDiff):
    memoryLength = sampler.settings['memoryLength']
    initialHessian = sampler.settings['initialHessian']
    noParameters = sampler.noParametersToEstimate
    identityMatrix = np.diag(np.ones(noParameters))
    inverseHessianEstimate = initialHessian**2 * identityMatrix
    noEffectiveSamples = 0

    for i in range(parametersDiff.shape[0]):
        differenceTerm = parametersDiff[i] - np.dot(inverseHessianEstimate, gradientsDiff[i])
        term1 = np.abs(parametersDiff[i], differenceTerm)
        term2 = np.linalg.norm(parametersDiff) * np.linalg.norm(differenceTerm)

        if term1 > sampler.settings['SR1UpdateLimit'] * term2:
            rankOneUpdate = np.outer(differenceTerm, differenceTerm) 
            rankOneUpdate /= np.dot(differenceTerm, gradientsDiff[i])
            inverseHessianEstimate += rankOneUpdate
            noEffectiveSamples += 1
    
    return inverseHessianEstimate, noEffectiveSamples