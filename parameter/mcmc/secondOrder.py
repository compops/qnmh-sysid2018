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
    model.storeParameters(mh.settings['initialParameters'])
    
    if model.areParametersValid():
        mh.logJacobian[0] = model.logJacobian()
        _, mh.logPrior[0] = model.logPrior()
        state.smoother(model)
        mh.logLikelihood[0] = state.logLikelihood
        mh.gradient[0, :] = state.gradientInternal
        mh.hessian[0, :, :] = correctHessian(state.hessianInternal)
        mh.states[0, :] = state.filteredStateEstimate
        mh.acceptProb[0] = 1.0
        mh.parametersUntransformed[0, :] = mh.settings['initialParameters']
        model.untransformParameters()
        mh.parameters[0, :] = model.getParameters()
    else:
        raise NameError("The initial values of the parameters does not result in a valid model.")

def proposeParameters(mh):
    noParameters = mh.settings['noParametersToEstimate']
    currentParameters = mh.parameters[mh.currentIteration - 1]
    currentGradient = mh.gradient[mh.currentIteration - 1, :]
    currentHessian = mh.hessian[mh.currentIteration - 1, :, :]

    stepSize = mh.settings['stepSize']
    proposalVariance = stepSize**2 * np.linalg.pinv(currentHessian)

    if (noParameters == 1):
        perturbation = stepSize / np.sqrt(currentHessian) * np.random.normal()
    else:
        perturbation = np.random.multivariate_normal(np.zeros(noParameters), proposalVariance)
    
    gradientContribution = 0.5 * stepSize**2 * np.dot(np.linalg.pinv(currentHessian), currentGradient)
    gradientContribution = np.asarray(gradientContribution).reshape(-1)
    mh.proposedParameters = currentParameters + gradientContribution + perturbation

    if mh.settings['verbose']:
        print("Proposing parameters: " + str(mh.proposedParameters) + " given " + str(currentParameters) + ".")


def computeAcceptanceProbability(mh, state, model):
    noParameters = mh.settings['noParametersToEstimate']
    stepSize = mh.settings['stepSize']

    currentParameters = mh.parametersUntransformed[mh.currentIteration - 1]
    currentLogJacobian = mh.logJacobian[mh.currentIteration - 1]
    currentLogPrior = mh.logPrior[mh.currentIteration - 1]
    currentLogLikelihood = mh.logLikelihood[mh.currentIteration - 1]
    currentGradient = mh.gradient[mh.currentIteration - 1, :]
    currentHessian = mh.hessian[mh.currentIteration - 1, :, :]
    currentStates = mh.states[mh.currentIteration - 1, :]
    
    if model.areParametersValid():
        proposedParameters = model.getParameters()
        proposedLogJacobian = model.logJacobian()
        _, proposedLogPrior = model.logPrior()
        state.smoother(model)
        proposedLogLikelihood = state.logLikelihood
        proposedGradient = state.gradientInternal
        proposedHessian = correctHessian(state.hessianInternal)
        proposedStates = state.filteredStateEstimate
        
        if isPositiveSemidefinite(proposedHessian):
            logPriorDifference = proposedLogPrior - currentLogPrior
            logLikelihoodDifference = proposedLogLikelihood - currentLogLikelihood

            proposalMean = currentParameters
            proposalMean += np.asarray(0.5 * stepSize**2 * np.dot(np.linalg.pinv(currentHessian), currentGradient)).reshape(-1)
            proposalVariance = stepSize**2 * np.linalg.pinv(currentHessian)
            logProposalProposed = multivariate_normal.logpdf(proposedParameters, proposalMean, proposalVariance)

            proposalMean = proposedParameters
            proposalMean += np.asarray(0.5 * stepSize**2 * np.dot(np.linalg.pinv(proposedHessian), proposedGradient)).reshape(-1)
            proposalVariance = stepSize**2 * np.linalg.pinv(proposedHessian)
            logProposalCurrent = multivariate_normal.logpdf(currentParameters, proposalMean, proposalVariance)

            logProposalDifference = logProposalProposed - logProposalCurrent
            logJacobianDifference = proposedLogJacobian - currentLogJacobian
            acceptProb = np.exp(logPriorDifference + logLikelihoodDifference + logProposalDifference + logJacobianDifference)
        else:
            acceptProb = 0.0

        if mh.settings['verbose']:
            print("proposedLogLikelihood: " + str(proposedLogLikelihood) + ".")
            print("logPriorDifference: " + str(logPriorDifference) + ".")
            print("logLikelihoodDifference: " + str(logLikelihoodDifference) + ".")
            print("logProposalDifference: " + str(logProposalDifference) + ".")
            print("acceptProb: " + str(acceptProb) + ".")            

        mh.proposedParametersUntransformed[mh.currentIteration, :] = proposedParameters
        mh.proposedLogJacobian[mh.currentIteration] = proposedLogJacobian
        mh.proposedLogPrior[mh.currentIteration] = proposedLogPrior
        mh.proposedLogLikelihood[mh.currentIteration] = proposedLogLikelihood
        mh.proposedStates[mh.currentIteration, :] = proposedStates
        mh.proposedGradient[mh.currentIteration, :] = proposedGradient
        mh.proposedHessian[mh.currentIteration, :, :] = proposedHessian
        mh.acceptProb[mh.currentIteration] = np.min((1.0, acceptProb))

    else:
        mh.currentAcceptanceProbability = 0.0

        if mh.settings['printWarningsForUnstableSystems']:
            print("Proposed parameters: " + str(mh.proposedParameters) + " results in an unstable system so rejecting.")
    
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
    
    return(x)