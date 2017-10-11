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

    if not 'memoryLength' in mh.settings: 
        mh.settings.update({'memoryLength': 10})
        print("Missing settings: memoryLength, defaulting to " + str(mh.settings['memoryLength']) + ".")   

    if not 'initialHessian' in mh.settings: 
        mh.settings.update({'initialHessian': 1e-8})
        print("Missing settings: initialHessian, defaulting to " + str(mh.settings['initialHessian']) + ".")           

    if not 'trustRegionSize' in mh.settings: 
        mh.settings.update({'trustRegionSize': 0})
        print("Missing settings: trustRegionSize, defaulting to " + str(mh.settings['trustRegionSize']) + ".")

    if not 'useDampedBFGS' in mh.settings: 
        mh.settings.update({'useDampedBFGS': True})
        print("Missing settings: useDampedBFGS, defaulting to " + str(mh.settings['useDampedBFGS']) + ".") 

def initialiseParameters(mh, state, model):
    model.storeParameters(mh.settings['initialParameters'])
    
    if model.areParametersValid():
        _, mh.logPrior[0] = model.logPrior()
        state.smoother(model)
        mh.logLikelihood[0] = state.logLikelihood
        mh.gradient[0, :] = state.gradientInternal
        mh.hessian[0, :, :], mh.naturalGradient[0, :] = estimateHessian(mh)
        mh.states[0, :] = state.filteredStateEstimate
        mh.acceptProb[0] = 1.0
        mh.parameters[0, :] = mh.settings['initialParameters']
    else:
        raise NameError("The initial values of the parameters does not result in a valid model.")

def proposeParameters(mh):
    noParameters = mh.settings['noParametersToEstimate']
    currentParameters = mh.parameters[mh.currentIteration - 1]
    currentNaturalGradient = mh.naturalGradient[mh.currentIteration - 1, :]
    currentHessian = mh.hessian[mh.currentIteration - 1, :, :]

    stepSize = mh.settings['stepSize']
    proposalVariance = stepSize * currentHessian

    if (noParameters == 1):
        perturbation = stepSize * currentHessian * np.random.normal()
    else:
        perturbation = np.matmul(proposalVariance, np.random.multivariate_normal(np.zeros(noParameters), np.diag(np.ones(noParameters))))
    
    gradientContribution = 0.5 * stepSize**2 * currentNaturalGradient
    mh.proposedParameters = currentParameters + gradientContribution + perturbation

    if mh.settings['verbose']:
        print("Proposing parameters: " + str(mh.proposedParameters) + " given " + str(currentParameters) + ".")

def computeAcceptanceProbability(mh, state, model):
    noParameters = mh.settings['noParametersToEstimate']
    stepSize = mh.settings['stepSize']

    currentParameters = mh.parameters[mh.currentIteration - 1]
    currentLogPrior = mh.logPrior[mh.currentIteration - 1]
    currentLogLikelihood = mh.logLikelihood[mh.currentIteration - 1]
    currentGradient = mh.gradient[mh.currentIteration - 1, :]
    currentHessian = mh.hessian[mh.currentIteration - 1, :, :]
    currentNaturalGradient = mh.naturalGradient[mh.currentIteration - 1, :]
    currentStates = mh.states[mh.currentIteration - 1, :]
    
    if model.areParametersValid():
        proposedParameters = mh.proposedParameters
        _, proposedLogPrior = model.logPrior()
        state.smoother(model)
        proposedLogLikelihood = state.logLikelihood
        proposedGradient = state.gradientInternal
        (proposedHessian, proposedNaturalGradient) = estimateHessian(mh)
        proposedStates = state.filteredStateEstimate
        
        if isPositiveSemidefinite(proposedHessian):
            logPriorDifference = proposedLogPrior - currentLogPrior
            logLikelihoodDifference = proposedLogLikelihood - currentLogLikelihood

            proposalMean = currentParameters
            proposalMean += 0.5 * stepSize**2 * currentNaturalGradient
            proposalVariance = stepSize * currentHessian
            logProposalProposed = multivariateGaussianLogPDF(proposedParameters, proposalMean, proposalVariance)

            proposalMean = proposedParameters
            proposalMean += 0.5 * stepSize**2 * proposedNaturalGradient
            proposalVariance = stepSize * proposedHessian
            logProposalCurrent = multivariateGaussianLogPDF(currentParameters, proposalMean, proposalVariance)

            logProposalDifference = logProposalProposed - logProposalCurrent

            acceptProb = np.exp(logPriorDifference + logLikelihoodDifference + logProposalDifference)
        else:
            acceptProb = 0.0

        if mh.settings['verbose']:
            print("proposedLogLikelihood: " + str(proposedLogLikelihood) + ".")
            print("logPriorDifference: " + str(logPriorDifference) + ".")
            print("logLikelihoodDifference: " + str(logLikelihoodDifference) + ".")
            print("logProposalDifference: " + str(logProposalDifference) + ".")
            print("acceptProb: " + str(acceptProb) + ".")            

        mh.proposedLogPrior = proposedLogPrior
        mh.proposedLogLikelihood = proposedLogLikelihood
        mh.proposedStates = proposedStates
        mh.proposedGradient = proposedGradient
        mh.proposedNaturalGradient = proposedNaturalGradient
        mh.proposedHessian = proposedHessian
        mh.currentAcceptanceProbability = np.min((1.0, acceptProb))
    else:
        mh.currentAcceptanceProbability = 0.0

        if mh.settings['printWarningsForUnstableSystems']:
            print("Proposed parameters: " + str(mh.proposedParameters) + " results in an unstable system so rejecting.")

def estimateHessian(mh):
    memoryLength = mh.settings['memoryLength']
    baseStepSize = mh.settings['baseStepSize']
    initialHessian = mh.settings['initialHessian']
    lam = mh.settings['trustRegionSize']

    noParameters = mh.noParametersToEstimate
    identityMatrix = np.diag(np.ones(noParameters))

    if mh.currentIteration < memoryLength:
        inverseHessianEstimate = identityMatrix * baseStepSize
        inverseHessianEstimateSquared = np.matmul(inverseHessianEstimate, inverseHessianEstimate.transpose())
        naturalGradient = np.dot(inverseHessianEstimateSquared, mh.gradient[mh.currentIteration - 1, :])
        return inverseHessianEstimate, naturalGradient
    
    # Extract parameters and gradidents
    idx = range(mh.currentIteration - memoryLength, mh.currentIteration)
    parameters = mh.parameters[idx, :]
    gradients = mh.gradient[idx, :]
    hessians = mh.hessian[idx, :, :]
    accepted = mh.accepted[idx]
    target = mh.logPrior[idx] + mh.logLikelihood[idx]

    # Keep only unique parameters and gradients
    idx = np.where(accepted > 0)[0]
    if len(idx) == 0:
        inverseHessianEstimate = identityMatrix * baseStepSize
        inverseHessianEstimateSquared = np.matmul(inverseHessianEstimate, inverseHessianEstimate.transpose())
        naturalGradient = np.dot(inverseHessianEstimateSquared, mh.gradient[mh.currentIteration - 1, :])
        return inverseHessianEstimate, naturalGradient

    parameters = parameters[idx, :]
    gradients = gradients[idx, :]
    hessians = hessians[idx, :, :]
    target = np.concatenate(target[idx]).reshape(-1)    
    accepted = accepted[idx, :]

    # Sort and compute differences
    idx = np.argsort(target)
    parameters = parameters[idx, :]
    gradients = gradients[idx, :]
    hessians = np.matmul(hessians[idx, :], hessians[idx, :])
    
    parametersDiff = np.zeros((len(idx) - 1, noParameters))
    gradientsDiff = np.zeros((len(idx) - 1, noParameters))

    # TODO: Uncertain about the sign of the gradients
    for i in range(len(idx) - 1):
        parametersDiff[i, :] = parameters[i+1, :] - parameters[i, :]
        gradientsDiff[i, :] = gradients[i+1, :] - gradients[i, :] + lam * np.matmul(hessians[i, :, :], gradients[i, :])
    
    inverseHessianEstimate = initialHessian * identityMatrix
    noEffectiveSamples = 0

    for i in range(parametersDiff.shape[0]):
        B = np.matmul(inverseHessianEstimate, inverseHessianEstimate.transpose())
        doUpdate = False

        if mh.settings['useDampedBFGS']:
            term1 = np.dot(parametersDiff[i], gradientsDiff[i])
            term2 = np.dot(np.dot(parametersDiff[i], B), parametersDiff[i])

            if term1 > 0.2 * term2:
                theta = 1.0
            else:
                theta = 0.5 * term2 / (term2 - term1)
            
            r = theta * gradientsDiff[i] + (1.0 - theta) * np.dot(B, parametersDiff[i])
            doUpdate = True
        else:
            if np.dot(parametersDiff[i], gradientsDiff[i]) > 0:
                doUpdate = True
                r = gradientsDiff[i]

        if doUpdate:
            quadraticFormSB = np.dot(np.dot(parametersDiff[i], B), parametersDiff[i])
            t = parametersDiff[i] / quadraticFormSB
            
            u1 = np.sqrt(quadraticFormSB / np.dot(parametersDiff[i], r))
            u2 = np.dot(B, parametersDiff[i])
            u = u1 * r + u2

            inverseHessianEstimate = np.matmul(identityMatrix - np.outer(u, t), inverseHessianEstimate)            
            noEffectiveSamples += 1
    
    mh.noEffectiveSamples[mh.currentIteration] = noEffectiveSamples

    inverseHessianEstimateSquared = np.matmul(inverseHessianEstimate, inverseHessianEstimate.transpose())
    naturalGradient = np.dot(inverseHessianEstimateSquared, mh.gradient[mh.currentIteration - 1, :])
    return inverseHessianEstimate, naturalGradient


def isPositiveSemidefinite(x):
    return np.all(np.linalg.eigvals(x) > 0)

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

    if not isPositiveSemidefinite(x):
        print("Failed fixing Hessian")
    
    return(x)

def multivariateGaussianLogPDF(x, mu, SigmaChol):
    term1 = -0.5 * len(mu) * np.log(2.0 * np.pi) 
    term2 = -1.0 * np.sum(np.log(np.diag(SigmaChol)))
    SigmaCholInverse = np.linalg.pinv(SigmaChol)
    SigmaInverse = np.matmul(SigmaCholInverse, SigmaCholInverse)
    term3 = -0.5 * np.dot(np.dot(x - mu, SigmaInverse), x - mu)
    
    return term1 + term2 + term3