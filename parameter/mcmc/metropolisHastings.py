import warnings
import numpy as np

from parameter.mcmc.helpers import truncateContribution
from parameter.mcmc.helpers import plotResults
from parameter.mcmc.helpers import checkSettings
from parameter.mcmc.helpers import printProgressReportToScreen
from parameter.mcmc.helpers import isHessianValid
from parameter.mcmc.helpers import logPDFGaussian

from parameter.mcmc.gradientEstimation import getGradient
from parameter.mcmc.gradientEstimation import getNaturalGradient
from parameter.mcmc.hessianEstimation import getHessian

warnings.filterwarnings("error")

class ParameterEstimator(object):
    #zvpost_linear = zvpost_linear_prototype
    #calcIACT = calcIACT_prototype
    #calcSJacobianD = calcSJacobianD_prototype
    #calcESS = calculateESS_prototype
    
    ###########################################################################
    def __init__(self, settings):
        self.settings = settings
        self.currentIteration = 0
        self.useGradientInformation = False
        self.useHesssianInformation = False

    ###########################################################################
    def initialise(self, model, samplerType):
        if samplerType is 'mh0':
            self.parameterEstimatorType = "Zero-order Metropolis-Hastings with Kalman methods"

        if samplerType is 'mh1':
            self.parameterEstimatorType = "First-order Metropolis-Hastings with Kalman methods"
            self.useGradientInformation = True

        if samplerType is 'mh2':
            self.parameterEstimatorType = "Second-order Metropolis-Hastings with Kalman methods"
            self.useGradientInformation = True
            self.useHesssianInformation = True         

        print("Sampling from the parameter posterior using: " + self.parameterEstimatorType)
        checkSettings(self)

        self.settings['noParametersToEstimate'] = model.noParametersToEstimate
        self.settings['noObservations'] = model.noObservations
        self.settings['parametersToEstimate'] = model.parametersToEstimate

        noIters = self.settings['noIters']
        noBurnInIters = self.settings['noBurnInIters']
        noParametersToEstimate = model.noParametersToEstimate
        noObservations = model.noObservations

        if noBurnInIters >= noIters:
            raise ValueError("metropolisHastings: noBurnInIters cannot be larger or equal to noIters.")

        self.noParametersToEstimate = noParametersToEstimate
        self.restrictedParameters = np.zeros((noIters, noParametersToEstimate))
        self.unrestrictedParameters = np.zeros((noIters, noParametersToEstimate))
        self.proposedRestrictedParameters = np.zeros((noIters, noParametersToEstimate))
        self.proposedUnrestrictedParameters = np.zeros((noIters, noParametersToEstimate))

        self.logPrior = np.zeros((noIters, 1))
        self.logLikelihood = np.zeros((noIters, 1))
        self.logJacobian = np.zeros((noIters, 1))
        self.states = np.zeros((noIters, noObservations))
        self.proposedLogPrior = np.zeros((noIters, 1))
        self.proposedLogLikelihood = np.zeros((noIters, 1))
        self.proposedLogJacobian = np.zeros((noIters, 1))
        self.proposedStates = np.zeros((noIters, noObservations))

        self.acceptProb = np.zeros((noIters, 1))
        self.accepted = np.zeros((noIters, 1))
        self.noEffectiveSamples = np.zeros((noIters, 1))
        
        self.gradient = np.zeros((noIters, noParametersToEstimate))
        self.naturalGradient = np.zeros((noIters, noParametersToEstimate))
        self.hessian = np.zeros((noIters, noParametersToEstimate, noParametersToEstimate))       

        self.proposedGradient = np.zeros((noIters, noParametersToEstimate))
        self.proposedNaturalGradient = np.zeros((noIters, noParametersToEstimate))
        self.proposedHessian = np.zeros((noIters, noParametersToEstimate, noParametersToEstimate))

        self.currentIteration = 0

    ###########################################################################
    def run(self, stateEstimator, model, samplerType):
        noIters = self.settings['noIters']
        noBurnInIters = self.settings['noBurnInIters']

        self.initialise(model, samplerType)
        self.initialiseParameters(stateEstimator, model)

        for iteration in range(1, noIters):
            self.currentIteration = iteration
            self.proposeParameters(model)
            self.computeAcceptanceProbability(stateEstimator, model)
            if (np.random.random(1) < self.acceptProb[iteration, :]):
                self.acceptParameters()
            else:
                self.rejectParameters()
            
            if self.settings['verbose']:
                print("Current unrestricted parameters: " + str(self.unrestrictedParameters[iteration, :]) + ".")
                print("Current restricted parameters: " + str(self.restrictedParameters[iteration, :]) + ".")
                input("Press ENTER to continue...")

            if np.remainder(iteration + 1, self.settings['nProgressReport']) == 0:
                printProgressReportToScreen(self)
            #self.printSimulationToFile()

        self.results = {}
        self.results.update({'parameterMeanEstimates': np.mean(self.restrictedParameters[noBurnInIters:noIters, :], axis=0)})
        self.results.update({'stateMeanEstimates': np.mean(self.states[noBurnInIters:noIters, :], axis=0)})
        self.results.update({'stateVarEstimates': np.var(self.states[noBurnInIters:noIters, :], axis=0)})
        self.results.update({'parameterTrace': self.restrictedParameters[noBurnInIters:noIters, :]})

    ###########################################################################
    def plot(self):
        plotResults(self)

    ###########################################################################
    def acceptParameters(self):
        i = self.currentIteration
        self.restrictedParameters[i, :] = self.proposedRestrictedParameters[i, :]
        self.unrestrictedParameters[i, :] = self.proposedUnrestrictedParameters[i, :]
        self.logJacobian[i] = self.proposedLogJacobian[i]
        self.logPrior[i, :] = self.proposedLogPrior[i, :]
        self.logLikelihood[i, :] = self.proposedLogLikelihood[i, :]
        self.states[i, :] = self.proposedStates[i, :]
        self.gradient[i, :] = self.proposedGradient[i, :]
        self.naturalGradient[i, :] = self.proposedNaturalGradient[i, :]
        self.hessian[i, :, :] = self.proposedHessian[i, :, :]
        self.accepted[i] = 1.0
    
    ###########################################################################
    def rejectParameters(self):
        i = self.currentIteration
        self.restrictedParameters[i, :] = self.restrictedParameters[i - 1, :]
        self.unrestrictedParameters[i, :] = self.unrestrictedParameters[i - 1, :]
        self.logJacobian[i] = self.logJacobian[i - 1]
        self.logPrior[i, :] = self.logPrior[i - 1, :]
        self.logLikelihood[i, :] = self.logLikelihood[i - 1, :]
        self.states[i, :] = self.states[i - 1, :]
        self.gradient[i, :] = self.gradient[i - 1, :]
        self.naturalGradient[i, :] = self.naturalGradient[i - 1, :]
        self.hessian[i, :, :] = self.hessian[i - 1, :, :]
        self.accepted[i] = 0.0

    ###########################################################################
    def initialiseParameters(self, state, model):
        model.storeRestrictedParameters(self.settings['initialParameters'])
        
        if model.areParametersValid():
            self.logJacobian[0] = model.logJacobian()
            _, self.logPrior[0] = model.logPrior()
            
            state.smoother(model)
            self.logLikelihood[0] = state.logLikelihood
            self.states[0, :] = state.filteredStateEstimate
            
            self.gradient[0, :] = getGradient(self, state)
            self.hessian[0, :, :] = getHessian(self, state)
            self.naturalGradient[0, :] = getNaturalGradient(self, self.gradient[0, :], self.hessian[0, :, :])

            self.restrictedParameters[0, :] = self.settings['initialParameters']
            self.unrestrictedParameters[0, :] = model.getUnrestrictedParameters()
            self.acceptProb[0] = 1.0
        else:
            raise NameError("The initial values of the parameters does not result in a valid model.")

    ###########################################################################
    def proposeParameters(self, model):
        noParameters = self.settings['noParametersToEstimate']
        currentUnrestrictedParameters = self.unrestrictedParameters[self.currentIteration - 1, :]
        currentNaturalGradient = self.naturalGradient[self.currentIteration - 1, :]
        currentHessian = self.hessian[self.currentIteration - 1, :, :]
        stepSize = self.settings['stepSize']

        if (noParameters == 1):
            perturbation = stepSize * np.sqrt(np.abs(currentHessian)) * np.random.normal()
        else:
            perturbation = np.random.multivariate_normal(np.zeros(noParameters), stepSize**2 * currentHessian)
        
        proposedUnrestrictedParameters = currentUnrestrictedParameters + currentNaturalGradient + perturbation
        if self.settings['verbose']:
            print("Proposing unrestricted parameters: " + str(proposedUnrestrictedParameters) + " given " + str(currentUnrestrictedParameters) + ".")

        model.storeUnrestrictedParameters(proposedUnrestrictedParameters)
        self.proposedUnrestrictedParameters[self.currentIteration, :] = model.getUnrestrictedParameters()
        self.proposedRestrictedParameters[self.currentIteration, :] = model.getRestrictedParameters()

    ###########################################################################
    def computeAcceptanceProbability(self, state, model):
        noParameters = self.settings['noParametersToEstimate']
        stepSize = self.settings['stepSize']

        currentRestrictedParameters = self.restrictedParameters[self.currentIteration - 1, :]
        currentUnrestrictedParameters = self.unrestrictedParameters[self.currentIteration - 1, :]
        currentLogJacobian = self.logJacobian[self.currentIteration - 1, :]
        currentLogPrior = self.logPrior[self.currentIteration - 1, :]
        currentLogLikelihood = self.logLikelihood[self.currentIteration - 1, :]
        currentNaturalGradient = self.naturalGradient[self.currentIteration - 1, :]
        currentHessian = self.hessian[self.currentIteration - 1, :, :]
        currentStates = self.states[self.currentIteration - 1, :]

        proposedRestrictedParameters = self.proposedRestrictedParameters[self.currentIteration, :]
        proposedUnrestrictedParameters = self.proposedUnrestrictedParameters[self.currentIteration, :]
        model.storeRestrictedParameters(proposedRestrictedParameters)

        if model.areParametersValid():
            proposedLogJacobian = model.logJacobian()
            _, proposedLogPrior = model.logPrior()
            
            state.smoother(model)
            proposedLogLikelihood = state.logLikelihood
            proposedStates = state.filteredStateEstimate

            proposedGradient = getGradient(self, state)
            proposedHessian = getHessian(self, state)
            proposedNaturalGradient = getNaturalGradient(self, proposedGradient, proposedHessian)
            
            if isHessianValid(proposedHessian):
                logPriorDifference = proposedLogPrior - currentLogPrior
                logLikelihoodDifference = proposedLogLikelihood - currentLogLikelihood

                proposalMean = currentUnrestrictedParameters + currentNaturalGradient
                proposalVariance = stepSize**2 * currentHessian
                #print(proposedUnrestrictedParameters)
                #print(proposalMean)
                #print(proposalVariance)
                logProposalProposed = logPDFGaussian(proposedUnrestrictedParameters, proposalMean, proposalVariance)

                proposalMean = proposedUnrestrictedParameters + proposedNaturalGradient
                proposalVariance = stepSize**2 * proposedHessian
                #print(currentRestrictedParameters)
                #print(proposalMean)
                #print(proposalVariance)
                logProposalCurrent = logPDFGaussian(currentRestrictedParameters, proposalMean, proposalVariance)

                logProposalDifference = logProposalProposed - logProposalCurrent
                logJacobianDifference = proposedLogJacobian - currentLogJacobian
                try:
                    acceptProb = np.exp(logPriorDifference + logLikelihoodDifference + logProposalDifference + logJacobianDifference)
                except:
                    if self.settings['verbose']:
                        print("Accepting as overflow occured.")
                    acceptProb = 1.0            
            else:
                print("Estimate of inverse Hessian is not PSD or is singular.")
                acceptProb = 0.0

            if self.settings['verbose'] and isHessianValid(proposedHessian):
                print("currentRestrictedParameters" + str(currentRestrictedParameters) + ".")
                print("proposedRestrictedParameters" + str(proposedRestrictedParameters) + ".")
                print("proposedLogLikelihood: " + str(proposedLogLikelihood) + ".")
                print("logPriorDifference: " + str(logPriorDifference) + ".")
                print("logLikelihoodDifference: " + str(logLikelihoodDifference) + ".")
                print("logProposalProposed: " + str(logProposalProposed) + ".")
                print("logProposalCurrent: " + str(logProposalCurrent) + ".")
                print("logProposalDifference: " + str(logProposalDifference) + ".")
                print("logJacobianDifference: " + str(logJacobianDifference) + ".")
                print("acceptProb: " + str(acceptProb) + ".")            

            self.acceptProb[self.currentIteration] = np.min((1.0, acceptProb))

            self.proposedLogJacobian[self.currentIteration] = proposedLogJacobian
            self.proposedLogPrior[self.currentIteration] = proposedLogPrior
            self.proposedLogLikelihood[self.currentIteration] = proposedLogLikelihood
            self.proposedStates[self.currentIteration, :] = proposedStates
            self.proposedNaturalGradient[self.currentIteration, :] = proposedNaturalGradient
            self.proposedGradient[self.currentIteration, :] = proposedGradient
            self.proposedHessian[self.currentIteration, :, :] = proposedHessian  
        else:          
            self.acceptProb[self.currentIteration] = 0.0
            print("Proposed parameters: " + str(proposedRestrictedParameters) + " results in an unstable system so rejecting.")


