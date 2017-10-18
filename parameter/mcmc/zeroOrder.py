import numpy as np

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
        mh.logJacobian[0] = model.logJacobian()
        _, mh.logPrior[0] = model.logPrior()
        state.filter(model)
        mh.logLikelihood[0] = state.logLikelihood
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

    stepSize = mh.settings['stepSize']
    if 'hessianEstimate' in mh.settings:
        hessianEstimate = mh.settings['hessianEstimate']
    else:
        hessianEstimate = np.diag(np.ones(noParameters))

    if (noParameters == 1):
        perturbation = stepSize * np.random.normal()
    else:
        perturbation = np.random.multivariate_normal(np.zeros(noParameters), stepSize**2 * hessianEstimate)
    
    mh.proposedParameters = currentParameters + perturbation

    if mh.settings['verbose']:
        print("Proposing parameters: " + str(mh.proposedParameters) + " given " + str(currentParameters) + ".")


def computeAcceptanceProbability(mh, state, model):
    currentLogJacobian = mh.logJacobian[mh.currentIteration - 1]
    currentLogPrior = mh.logPrior[mh.currentIteration - 1]
    currentLogLikelihood = mh.logLikelihood[mh.currentIteration - 1]
    currentStates = mh.states[mh.currentIteration - 1, :]
    
    if model.areParametersValid():
        _, proposedLogPrior = model.logPrior()
        proposedLogJacobian = model.logJacobian()
        state.filter(model)
        proposedLogLikelihood = state.logLikelihood
        proposedStates = state.filteredStateEstimate
        
        logPriorDifference = proposedLogPrior - currentLogPrior
        logLikelihoodDifference = proposedLogLikelihood - currentLogLikelihood
        
        logProposalDifference = 0.0
        logJacobianDifference = proposedLogJacobian - currentLogJacobian
        acceptProb = np.exp(logPriorDifference + logLikelihoodDifference + logProposalDifference + logJacobianDifference)

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
        mh.acceptProb[mh.currentIteration] = np.min((1.0, acceptProb))
    else:
        mh.currentAcceptanceProbability = 0.0

        if mh.settings['printWarningsForUnstableSystems']:
            print("Proposed parameters: " + str(mh.proposedParameters) + " results in an unstable system so rejecting.")
    
