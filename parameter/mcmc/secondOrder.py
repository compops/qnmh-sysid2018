import warnings
warnings.filterwarnings("error")

import numpy as np
from scipy.stats import multivariate_normal

def checkSettings(mh):
    if not 'noIters' in mh.settings:
        mh.settings.update({'noIters': 1000})
        print("Missing settings: noItermh, defaulting to " + str(mh.settings['noIters']) + ".")

    if not 'noBurnInIters' in mh.settings:
        mh.settings.update({'noBurnInIters': 250})
        print("Missing settings: noBurnInItermh, defaulting to " + str(mh.settings['noBurnInIters']) + ".")

    if not 'stepSize' in mh.settings:
        mh.settings.update({'stepSize': 1.0})
        print("Missing settings: stepSize, defaulting to " + str(mh.settings['stepSize']) + ".")   

    if not 'iterationsBetweenProgressReports' in mh.settings: 
        mh.settings.update({'nProgressReport': 100})
        print("Missing settings: nProgressReport, defaulting to " + str(mh.settings['nProgressReport']) + ".")   

    if not 'printWarningsForUnstableSystems' in mh.settings: 
        mh.settings.update({'printWarningsForUnstableSystems': False})
        print("Missing settings: printWarningsForUnstableSystemmh, defaulting to " + str(mh.settings['printWarningsForUnstableSystems']) + ".")   

def initialiseParameters(mh, state, model):

    model.storeRestrictedParameters(mh.settings['initialParameters'])
    
    if model.areParametersValid():
        mh.logJacobian[0] = model.logJacobian()
        _, mh.logPrior[0] = model.logPrior()
        state.smoother(model)
        mh.logLikelihood[0] = state.logLikelihood
        mh.gradient[0, :] = state.gradientInternal
        mh.states[0, :] = state.filteredStateEstimate
        mh.acceptProb[0] = 1.0
        mh.hessian[0, :, :] = getHessian(mh, state)

        mh.restrictedParameters[0, :] = mh.settings['initialParameters']
        mh.unrestrictedParameters[0, :] = model.getUnrestrictedParameters()
    else:
        raise NameError("The initial values of the parameters does not result in a valid model.")

def proposeParameters(mh, model):
    noParameters = mh.settings['noParametersToEstimate']
    currentUnrestrictedParameters = mh.unrestrictedParameters[mh.currentIteration - 1, :]
    currentGradient = mh.gradient[mh.currentIteration - 1, :]
    currentHessian = mh.hessian[mh.currentIteration - 1, :, :]
    stepSize = mh.settings['stepSize']

    if (noParameters == 1):
        perturbation = stepSize * np.sqrt(np.abs(currentHessian)) * np.random.normal()
    else:
        perturbation = np.random.multivariate_normal(np.zeros(noParameters), stepSize**2 * correctHessian((currentHessian)))

    if mh.currentIteration > mh.settings['memoryLength']:
        gradientContribution = np.asarray(0.5 * stepSize**2 * np.dot(currentHessian, currentGradient)).reshape(-1)
    else:
        gradientContribution = 0.0
    
    proposedUnrestrictedParameters = currentUnrestrictedParameters + gradientContribution + perturbation
    
    model.storeUnrestrictedParameters(proposedUnrestrictedParameters)
    mh.proposedUnrestrictedParameters[mh.currentIteration, :] = model.getUnrestrictedParameters()
    mh.proposedRestrictedParameters[mh.currentIteration, :] = model.getRestrictedParameters()

    if mh.settings['verbose']:
        print("Proposing unrestricted parameters: " + str(proposedUnrestrictedParameters) + " given " + str(currentUnrestrictedParameters) + ".")


def computeAcceptanceProbability(mh, state, model):
    noParameters = mh.settings['noParametersToEstimate']
    stepSize = mh.settings['stepSize']

    currentRestrictedParameters = mh.restrictedParameters[mh.currentIteration - 1, :]
    currentLogJacobian = mh.logJacobian[mh.currentIteration - 1, :]
    currentLogPrior = mh.logPrior[mh.currentIteration - 1, :]
    currentLogLikelihood = mh.logLikelihood[mh.currentIteration - 1, :]
    currentGradient = mh.gradient[mh.currentIteration - 1, :]
    currentHessian = mh.hessian[mh.currentIteration - 1, :, :]
    currentStates = mh.states[mh.currentIteration - 1, :]

    proposedRestrictedParameters = mh.proposedRestrictedParameters[mh.currentIteration, :]
    model.storeRestrictedParameters(proposedRestrictedParameters)

    if model.areParametersValid():
        proposedLogJacobian = model.logJacobian()
        _, proposedLogPrior = model.logPrior()
        state.smoother(model)
        proposedLogLikelihood = state.logLikelihood
        proposedGradient = state.gradientInternal
        proposedHessian = getHessian(mh, state)
        proposedStates = state.filteredStateEstimate
        
        #if isPositiveSemidefinite(proposedHessian):
        if True:
            logPriorDifference = proposedLogPrior - currentLogPrior
            logLikelihoodDifference = proposedLogLikelihood - currentLogLikelihood

            proposalMean = currentRestrictedParameters
            if mh.currentIteration > mh.settings['memoryLength']:
                proposalMean += np.asarray(0.5 * stepSize**2 * np.dot(currentHessian, currentGradient)).reshape(-1)
            proposalVariance = stepSize**2 * correctHessian(currentHessian)
            logProposalProposed = multivariate_normal.logpdf(proposedRestrictedParameters, proposalMean, proposalVariance)

            proposalMean = proposedRestrictedParameters
            if mh.currentIteration > mh.settings['memoryLength']:
                proposalMean += np.asarray(0.5 * stepSize**2 * np.dot(proposedHessian, proposedGradient)).reshape(-1)
            proposalVariance = stepSize**2 * correctHessian(proposedHessian)
            logProposalCurrent = multivariate_normal.logpdf(currentRestrictedParameters, proposalMean, proposalVariance)

            logProposalDifference = logProposalProposed - logProposalCurrent
            logJacobianDifference = proposedLogJacobian - currentLogJacobian

            try:
                acceptProb = np.exp(logPriorDifference + logLikelihoodDifference + logProposalDifference + logJacobianDifference)
            except:
                if mh.settings['verbose']:
                    print("Rejecting as overflow occured.")
                acceptProb = 0.0            
        else:
            print("Estimate of inverse Hessian is not PSD.")
            acceptProb = 0.0

        if mh.settings['verbose']:
            print("currentRestrictedParameters" + str(currentRestrictedParameters) + ".")
            print("proposedRestrictedParameters" + str(proposedRestrictedParameters) + ".")
            print("proposedLogLikelihood: " + str(proposedLogLikelihood) + ".")
            print("logPriorDifference: " + str(logPriorDifference) + ".")
            print("logLikelihoodDifference: " + str(logLikelihoodDifference) + ".")
            print("logProposalDifference: " + str(logProposalDifference) + ".")
            print("acceptProb: " + str(acceptProb) + ".")            

        mh.acceptProb[mh.currentIteration] = np.min((1.0, acceptProb))

        mh.proposedLogJacobian[mh.currentIteration] = proposedLogJacobian
        mh.proposedLogPrior[mh.currentIteration] = proposedLogPrior
        mh.proposedLogLikelihood[mh.currentIteration] = proposedLogLikelihood
        mh.proposedStates[mh.currentIteration, :] = proposedStates
        mh.proposedGradient[mh.currentIteration, :] = proposedGradient
        mh.proposedHessian[mh.currentIteration, :, :] = proposedHessian  
    else:          
        mh.acceptProb[mh.currentIteration] = 0.0
        print("Proposed parameters: " + str(proposedRestrictedParameters) + " results in an unstable system so rejecting.")

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

def isPositiveSemidefinite(x):
    return np.all(np.linalg.eigvals(x) > 0)

def getHessian(mh, state):
    if mh.settings['hessianEstimate'] is 'kalman':
        return np.linalg.inv(state.hessianInternal)
    if mh.settings['hessianEstimate'] is 'BFGS':            
        return estimateHessianBFGS(mh)
    if mh.settings['hessianEstimate'] is 'SR1':
        return estimateHessianSR1(mh)

def correctHessian(x, approach='regularise'):

    if not isPositiveSemidefinite(x):
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


def estimateHessianBFGS(mh, useInformationFromRejectedSteps=False):
    memoryLength = mh.settings['memoryLength']
    baseStepSize = mh.settings['baseStepSize']
    initialHessian = mh.settings['initialHessian']

    noParameters = mh.noParametersToEstimate
    identityMatrix = np.diag(np.ones(noParameters))

    # Not enough iterations have been carried out, defBFGSault to diagonal Hessian
    if mh.currentIteration < memoryLength:
        return identityMatrix * baseStepSize**2
    
    # Extract parameters and gradients
    idx = range(mh.currentIteration - memoryLength, mh.currentIteration)
    parameters = mh.proposedParameters[idx, :]
    gradients = mh.proposedGradient[idx, :]
    hessians = mh.proposedHessian[idx, :, :]
    accepted = mh.accepted[idx]
    target = np.concatenate(mh.proposedLogPrior[idx] + mh.proposedLogLikelihood[idx]).reshape(-1)

    # Keep only unique parameters and gradients
    if not useInformationFromRejectedSteps:
        idx = np.where(accepted > 0)[0]

        # No available infomation, so quit
        if len(idx) == 0:
            return identityMatrix * baseStepSize**2
        
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
    
    # Initialisation of H0
    if initialHessian is 'scaledProposedGradient':
        inverseHessianEstimate = identityMatrix * baseStepSize / np.linalg.norm(proposedGradient, 2)

    if initialHessian is 'scaledCurvature':
        scaledCurvature = np.dot(parametersDiff[0], gradientsDiff[0]) * np.dot(gradientsDiff[0], gradientsDiff[0])
        inverseHessianEstimate = identityMatrix * np.abs(scaledCurvature)

    if isinstance(initialHessian, float):
        inverseHessianEstimate = initialHessian**2 * identityMatrix
    
    noEffectiveSamples = 0

    for i in range(parametersDiff.shape[0]):
        doUpdate = False

        if mh.settings['useDampedBFGS']:
            term1 = np.dot(parametersDiff[i], gradientsDiff[i])
            term2 = np.dot(np.dot(parametersDiff[i], B), parametersDiff[i])

            if (term1 > 0.2 * term2):
                theta = 1.0
            else:
                theta = 0.8 * term2 / (term2 - term1)
            
            r = theta * gradientsDiff[i] + (1.0 - theta) * np.dot(B, parametersDiff[i])
            doUpdate = True
        else:
            if np.dot(parametersDiff[i], gradientsDiff[i]) > 0:
                doUpdate = True
                r = gradientsDiff[i]

        if doUpdate:
            quadraticFormSB = np.dot(np.dot(parametersDiff[i], B), parametersDiff[i])
            curvatureCondition = np.dot(parametersDiff[i], r)

            term1 = (curvatureCondition + quadraticFormSB) / curvatureCondition**2
            term1 *= np.outer(parametersDiff[i], parametersDiff[i])

            term2 = np.dot(np.dot(inverseHessianEstimate, r), parametersDiff[i])
            term2 += np.dot(np.dot(parametersDiff[i], r), inverseHessianEstimate)
            term2 /= curvatureCondition

            inverseHessianEstimate += term1 - term2
            noEffectiveSamples += 1
    
    mh.noEffectiveSamples[mh.currentIteration] = noEffectiveSamples
    naturalGradient = np.dot(inverseHessianEstimate, proposedGradient)
    return inverseHessianEstimate, naturalGradient


def estimateHessianSR1(mh, useInformationFromRejectedSteps=False):
    memoryLength = mh.settings['memoryLength']
    baseStepSize = mh.settings['baseStepSize']
    initialHessian = mh.settings['initialHessian']

    noParameters = mh.noParametersToEstimate
    identityMatrix = np.diag(np.ones(noParameters))

    # Not enough iterations have been carried out, default to diagonal Hessian
    if mh.currentIteration < memoryLength:
        return identityMatrix * baseStepSize**2
    
    # Extract parameters and gradients
    idx = range(mh.currentIteration - memoryLength, mh.currentIteration)
    parameters = mh.proposedParameters[idx, :]
    gradients = mh.proposedGradient[idx, :]
    hessians = mh.proposedHessian[idx, :, :]
    accepted = mh.accepted[idx]
    target = np.concatenate(mh.proposedLogPrior[idx] + mh.proposedLogLikelihood[idx]).reshape(-1)

    # Keep only unique parameters and gradients
    if not useInformationFromRejectedSteps:
        idx = np.where(accepted > 0)[0]

        # No available infomation, so quit
        if len(idx) == 0:
            return identityMatrix * baseStepSize**2
        
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
    
    # Initialisation of H0
    inverseHessianEstimate = initialHessian**2 * identityMatrix
    
    noEffectiveSamples = 0

    for i in range(parametersDiff.shape[0]):
        differenceTerm = parametersDiff[i] - np.dot(inverseHessianEstimate, gradientsDiff[i])
        term1 = np.abs(parametersDiff[i], differenceTerm)
        term2 = np.linalg.norm(parametersDiff) * np.linalg.norm(differenceTerm)

        if term1 > 1e-8 * term2:
            rankOneUpdate = np.outer(differenceTerm, differenceTerm) 
            rankOneUpdate /= np.dot(differenceTerm, gradientsDiff[i])
            inverseHessianEstimate += rankOneUpdate
            noEffectiveSamples += 1
    
    mh.noEffectiveSamples[mh.currentIteration] = noEffectiveSamples
    return np.abs(inverseHessianEstimate)