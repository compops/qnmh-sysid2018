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
        mh.settings.update({'stepSize': 0.10})
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
        _, mh.logPrior[0] = model.logPrior()
        state.smoother(model)
        mh.logLikelihood[0] = state.logLikelihood
        mh.gradient[0, :] = state.gradientInternal
        mh.states[0, :] = state.filteredStateEstimate
        mh.acceptProb[0] = 1.0
        mh.parameters[0, :] = mh.settings['initialParameters']
    else:
        raise NameError("The initial values of the parameters does not result in a valid model.")

def proposeParameters(mh):
    noParameters = mh.settings['noParametersToEstimate']
    currentParameters = mh.parameters[mh.currentIteration - 1]
    currentGradient = mh.gradient[mh.currentIteration - 1]

    stepSize = mh.settings['stepSize']
    if 'hessianEstimate' in mh.settings:
        proposalVariance = stepSize**2 * mh.settings['hessianEstimate']
    else:
        proposalVariance = stepSize**2 * np.diag(np.ones(noParameters))

    if (noParameters == 1):
        perturbation = stepSize * np.random.normal()
    else:
        perturbation = np.random.multivariate_normal(np.zeros(noParameters), proposalVariance)
    
    mh.proposedParameters = currentParameters + 0.5 * stepSize**2 * currentGradient + perturbation

    if mh.settings['verbose']:
        print("Proposing parameters: " + str(mh.proposedParameters) + " given " + str(currentParameters) + ".")


def computeAcceptanceProbability(mh, state, model):
    noParameters = mh.settings['noParametersToEstimate']
    stepSize = mh.settings['stepSize']

    if 'hessianEstimate' in mh.settings:
        proposalVariance = stepSize**2 * mh.settings['hessianEstimate']
    else:
        proposalVariance = stepSize**2 * np.diag(np.ones(noParameters))

    currentParameters = mh.parameters[mh.currentIteration - 1]
    currentLogPrior = mh.logPrior[mh.currentIteration - 1]
    currentLogLikelihood = mh.logLikelihood[mh.currentIteration - 1]
    currentGradient = mh.gradient[mh.currentIteration - 1]
    currentStates = mh.states[mh.currentIteration - 1, :]
    
    if model.areParametersValid():
        proposedParameters = mh.proposedParameters
        _, proposedLogPrior = model.logPrior()
        state.smoother(model)
        proposedLogLikelihood = state.logLikelihood
        proposedGradient = state.gradientInternal
        proposedStates = state.filteredStateEstimate
        
        logPriorDifference = proposedLogPrior - currentLogPrior
        logLikelihoodDifference = proposedLogLikelihood - currentLogLikelihood

        mean = currentParameters + 0.5 * stepSize**2 * currentGradient
        logProposalProposed = multivariate_normal.logpdf(proposedParameters, mean, proposalVariance)

        mean = proposedParameters + 0.5 * stepSize**2 * proposedGradient
        logProposalCurrent = multivariate_normal.logpdf(currentParameters, mean, proposalVariance)

        logProposalDifference = logProposalProposed - logProposalCurrent

        acceptProb = np.exp(logPriorDifference + logLikelihoodDifference + logProposalDifference)

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
        mh.currentAcceptanceProbability = np.min((1.0, acceptProb))

    else:
        mh.proposedLogPrior = currentLogPrior
        mh.proposedLogLikelihood = currentLogLikelihood
        mh.proposedStates = currentStates
        mh.proposedGradient = currentGradient
        mh.currentAcceptanceProbability = 0.0

        if mh.settings['printWarningsForUnstableSystems']:
            print("Proposed parameters: " + str(mh.proposedParameters) + " results in an unstable system so rejecting.")
    
