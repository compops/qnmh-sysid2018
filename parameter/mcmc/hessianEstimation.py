import numpy as np
from parameter.mcmc.helpers import isPositiveSemiDefinite

def getHessian(sampler, stateEstimator, proposedGradient):
    inverseHessian = np.eye(sampler.settings['noParametersToEstimate']) 
    inverseHessian *= sampler.settings['initialHessian']**2
    inverseHessian *+ sampler.settings['stepSize']**2

    if sampler.useHesssianInformation:
        if sampler.settings['hessianEstimate'] is 'Kalman':
            hessianEstimate = sampler.settings['stepSize']**2 * np.linalg.inv(stateEstimator.hessianInternal)
            return correctHessian(hessianEstimate, sampler)
        elif sampler.currentIteration > sampler.settings['memoryLength']:
            return sampler.settings['stepSize']**2 * estimateHessianQN(sampler, proposedGradient)
    
    if sampler.settings['verbose']:
        print("Current inverseHessian: " + str(inverseHessian) + ".")    
    return inverseHessian

def correctHessian(x, sampler):
    approach = sampler.settings['hessianCorrectionApproach']
    # No correction
    if not approach:
        return(x)
    
    if isinstance(x, bool) or not isPositiveSemiDefinite(x):
        sampler.noCorrectedHessianEstimates +=1
        sampler.hessianEstimateCorrectedAtIteration.append(sampler.currentIteration)

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

def estimateHessianQN(sampler, proposedGradient):
    memoryLength = sampler.settings['memoryLength']
    initHessian = sampler.settings['initialHessian']
    approach = sampler.settings['hessianEstimate']
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
                return identityMatrix * initHessian**2
        
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

    initHessian = initialiseHessianEstimate(sampler, proposedGradient, parametersDiff, gradientsDiff)

    if approach is 'BFGS':
        hessianEstimate, noSamples = estimateHessianBFGS(initHessian, sampler, parametersDiff, gradientsDiff, curvatureCondtion=True)
        hessianEstimate = correctHessian(hessianEstimate, sampler)

    elif approach is 'BFGSdamped':
        hessianEstimate, noSamples = estimateHessianBFGS(initHessian, sampler, parametersDiff, gradientsDiff, curvatureCondtion='damped')
    
    elif approach is 'BFGSignoreCurvatureCondition':
        hessianEstimate, noSamples = estimateHessianBFGS(initHessian, sampler, parametersDiff, gradientsDiff, curvatureCondtion='ignore')
        hessianEstimate = correctHessian(hessianEstimate, sampler)
    
    elif approach is 'SR1':
        hessianEstimate, noSamples = estimateHessianSR1(initHessian, sampler, parametersDiff, gradientsDiff)
        hessianEstimate = correctHessian(hessianEstimate, sampler)

    else:
        raise NameError("Unknown quasi-Newton algorithm selected...")

    sampler.noEffectiveSamples[sampler.currentIteration] = noSamples
    return hessianEstimate

def estimateHessianBFGS(hessianEstimate, sampler, parametersDiff, gradientsDiff, curvatureCondtion=True):
    memoryLength = sampler.settings['memoryLength']
    noSamples = 0
    violationsCurvatureCondition = 0

    for i in range(parametersDiff.shape[0]):
        doUpdate = False

        if curvatureCondtion is True:
            if np.dot(parametersDiff[i], gradientsDiff[i]) < 0.0:
                doUpdate = True
                r = gradientsDiff[i]
            else: 
                violationsCurvatureCondition +=1
        
        elif curvatureCondtion is 'damped':
            term1 = np.dot(parametersDiff[i], gradientsDiff[i])
            term2 = np.dot(np.dot(parametersDiff[i], np.linalg.inv(hessianEstimate)), parametersDiff[i])
            if (term1 > 0.2 * term2):
                theta = 1.0
            else:
                theta = 0.8 * term2 / (term2 - term1)
            
            r = theta * gradientsDiff[i] + (1.0 - theta) * np.dot(np.linalg.inv(hessianEstimate), parametersDiff[i])
            doUpdate = True
        elif curvatureCondtion is 'ignore':
            doUpdate = True
            r = gradientsDiff[i]
        else:
            raise NameError("Unknown flag curvatureCondtion given to function")

        if doUpdate:
            noSamples += 1
            rho = 1.0 / np.dot(r, parametersDiff[i])
            term1 = np.eye(len(parametersDiff[i])) - rho * np.outer(parametersDiff[i], r)
            term2 = np.eye(len(parametersDiff[i])) - rho * np.outer(r, parametersDiff[i])
            term3 = rho * np.outer(parametersDiff[i], parametersDiff[i])
            hessianEstimate = np.matmul(np.matmul(term1, hessianEstimate), term2) + term3
    
    #print("BFGS, noMaxSamples: " + str(len(parametersDiff)) + ", noSamples: " + str(noSamples) + " and violationsCurvatureCondition: " + str(violationsCurvatureCondition) + ".")
    return -hessianEstimate, noSamples

def estimateHessianSR1(hessianEstimate, sampler, parametersDiff, gradientsDiff):
    memoryLength = sampler.settings['memoryLength']
    initHessian = sampler.settings['initialHessian']
    noParameters = sampler.noParametersToEstimate
    identityMatrix = np.diag(np.ones(noParameters))
    noSamples = 0

    for i in range(parametersDiff.shape[0]):
        differenceTerm = parametersDiff[i] - np.dot(hessianEstimate, gradientsDiff[i])
        if np.dot(differenceTerm, gradientsDiff[i]) != 0.0:
            rankOneUpdate = np.outer(differenceTerm, differenceTerm) 
            rankOneUpdate /= np.dot(differenceTerm, gradientsDiff[i])
            hessianEstimate += rankOneUpdate
            noSamples += 1
    
    return -hessianEstimate, noSamples

def initialiseHessianEstimate(sampler, proposedGradient, parametersDiff, gradientsDiff):
    approach = sampler.settings['hessianEstimateAdaptInitialisation']
    noParameters = sampler.noParametersToEstimate
    identityMatrix = np.diag(np.ones(noParameters))

    if approach is False:
        return sampler.settings['initialHessian']**2 * identityMatrix

    if approach is 'scaledProposedGradient':
        return identityMatrix * sampler.settings['initialHessian']**2 / np.linalg.norm(proposedGradient, 2)

    if approach is 'scaledCurvature':
        scaledCurvature = np.dot(parametersDiff[0], gradientsDiff[0]) * np.dot(gradientsDiff[0], gradientsDiff[0])
        return identityMatrix * np.abs(scaledCurvature)

    