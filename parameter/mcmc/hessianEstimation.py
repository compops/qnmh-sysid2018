import numpy as np
from parameter.mcmc.helpers import isPositiveSemiDefinite

def getHessian(sampler, stateEstimator):
    inverseHessian = np.eye(sampler.settings['noParametersToEstimate']) 
    inverseHessian *= sampler.settings['initialHessian']**2
    inverseHessian *+ sampler.settings['stepSize']**2

    if sampler.useHesssianInformation:
        if sampler.settings['hessianEstimate'] is 'kalman':
            hessianEstimate = sampler.settings['stepSize']**2 * np.linalg.inv(stateEstimator.hessianInternal)
            return correctHessian(hessianEstimate, sampler)
        elif sampler.currentIteration > sampler.settings['memoryLength']:
            return sampler.settings['stepSize']**2 * estimateHessianQN(sampler)

    
    if sampler.settings['verbose']:
        print("Current inverseHessian: " + str(inverseHessian) + ".")    
    return inverseHessian

def correctHessian(x, sampler):
    approach = sampler.settings['hessianCorrectionApproach']
    # No correction
    if not approach:
        return(x)
    
    if isinstance(x, bool) or not isPositiveSemiDefinite(x):

        if isPositiveSemiDefinite(-x):
            print("Iteration: " + str(sampler.currentIteration) +  ", switched to negative Hessian estimate...")
            return -x
        
        if approach is 'replace':
            if sampler.currentIteration > sampler.settings['noBurnInIters']:
                if sampler.settings['verbose'] or sampler.settings['informOfHessianCorrection']:
                    print("Iteration: " + str(sampler.currentIteration) +  ", corrected Hessian by replacing with estimate from latter half of burn-in.")
                
                if not hasattr(sampler, 'empericalHessianEstimate'):
                    idx = range(int(0.5 * sampler.settings['noBurnInIters']), sampler.settings['noBurnInIters'])
                    sampler.empericalHessianEstimate = np.cov(sampler.unrestrictedParameters[idx, :], rowvar=False)
                    print("Iteration: " + str(sampler.currentIteration) +  ", computed an empirical estimate of the posterior covariance to replace ND Hessian estimates.")
                return(sampler.empericalHessianEstimate)
            else:
                return np.diag(np.ones(sampler.noParametersToEstimate)) * sampler.settings['initialHessian']**2
        
        # Add a diagonal matrix proportional to the largest negative eigenvalue
        elif approach is 'regularise':
                smallestEigenValue = np.min(np.linalg.eig(x)[0])
                if sampler.settings['verbose'] or sampler.settings['informOfHessianCorrection']:
                    print("Iteration: " + str(sampler.currentIteration) +  ", corrected Hessian by adding diagonal matrix with elements: " + str(-2.0 * smallestEigenValue))
                return x - 2.0 * smallestEigenValue * np.eye(x.shape[0])

        # Flip the negative eigenvalues
        elif approach is 'flip':
                if sampler.settings['verbose'] or sampler.settings['informOfHessianCorrection']:
                    print("Iteration: " + str(sampler.currentIteration) +  ", corrected Hessian by flipping negative eigenvalues to positive.")
                evDecomp = np.linalg.eig(x)
                return np.dot(np.dot(evDecomp[1], np.diag(np.abs(evDecomp[0]))), evDecomp[1])
        else:
            raise ValueError("Unknown Hessian correction strategy...")
    else:
        return x

def estimateHessianQN(sampler):
    memoryLength = sampler.settings['memoryLength']
    initialHessian = sampler.settings['initialHessian']
    method = sampler.settings['hessianEstimate']
    useOnlyInformationFromAcceptedSteps = sampler.settings['hessianEstimateOnlyAcceptedInformation']
    noParameters = sampler.noParametersToEstimate
    identityMatrix = np.diag(np.ones(noParameters))
   
    # Extract parameters and gradients
    idx = range(sampler.currentIteration - memoryLength, sampler.currentIteration)
    parameters = sampler.proposedUnrestrictedParameters[idx, :]
    gradients = sampler.proposedGradient[idx, :]
    hessians = sampler.proposedHessian[idx, :, :]
    accepted = sampler.accepted[idx]
    target = np.concatenate(sampler.proposedLogPrior[idx] + sampler.proposedLogLikelihood[idx]).reshape(-1)

    # Keep only unique parameters and gradients
    if useOnlyInformationFromAcceptedSteps:
        idx = np.where(accepted > 0)[0]

        # No available infomation, so quit
        if len(idx) == 0:
            if sampler.settings['verbose']:
                print("Not enough samples to estimate Hessian...")
            if sampler.settings['hessianCorrectionApproach'] is 'replace':
                return correctHessian(True, sampler)
            else:    
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

    elif method is 'BFGS':
        inverseHessianEstimate, noEffectiveSamples = estimateHessianBFGS(sampler, parametersDiff, gradientsDiff)
        inverseHessianEstimate = correctHessian(inverseHessianEstimate, sampler)
    
    elif method is 'SR1':
        inverseHessianEstimate, noEffectiveSamples = estimateHessianSR1(sampler, parametersDiff, gradientsDiff)
        inverseHessianEstimate = correctHessian(inverseHessianEstimate, sampler)

    else:
        raise NameError("Unknown quasi-Newton algorithm selected...")

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
        if np.dot(differenceTerm, gradientsDiff[i]) != 0.0:
            rankOneUpdate = np.outer(differenceTerm, differenceTerm) 
            rankOneUpdate /= np.dot(differenceTerm, gradientsDiff[i])
            inverseHessianEstimate += rankOneUpdate
            noEffectiveSamples += 1
    
    return -inverseHessianEstimate, noEffectiveSamples