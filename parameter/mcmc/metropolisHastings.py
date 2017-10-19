import pandas
import numpy as np
import parameter.mcmc.helpers as helpers

class ParameterEstimator(object):
    #zvpost_linear = zvpost_linear_prototype
    #calcIACT = calcIACT_prototype
    #calcSJacobianD = calcSJacobianD_prototype
    #calcESS = calculateESS_prototype
    #     
    def __init__(self, settings):
        self.settings = settings
        self.currentIteration = 0

    def run(self, stateEstimator, model, samplerType):

        if samplerType is 'mh0':
            self.parameterEstimatorType = "Zero-order Metropolis-Hastings with Kalman methods"
            import parameter.mcmc.zeroOrder as samplingHelpers

        if samplerType is 'mh1':
            self.parameterEstimatorType = "First-order Metropolis-Hastings with Kalman methods"
            import parameter.mcmc.firstOrder as samplingHelpers

        if samplerType is 'mh2':
            self.parameterEstimatorType = "Second-order Metropolis-Hastings with Kalman methods"
            import parameter.mcmc.secondOrder as samplingHelpers

        if samplerType is 'qmh':
            self.parameterEstimatorType = "Second-order Metropolis-Hastings using L-BFGS with Kalman methods"
            import parameter.mcmc.quasiNewtonBFGS as samplingHelpers            

        print("Sampling from the parameter posterior using: " + self.parameterEstimatorType)
        samplingHelpers.checkSettings(self)

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
        self.hessian = np.zeros((noIters, noParametersToEstimate, noParametersToEstimate))       

        self.proposedGradient = np.zeros((noIters, noParametersToEstimate))
        self.proposedHessian = np.zeros((noIters, noParametersToEstimate, noParametersToEstimate))

        self.currentIteration = 0
        samplingHelpers.initialiseParameters(self, stateEstimator, model)

        for iteration in range(1, noIters):
            self.currentIteration = iteration

            samplingHelpers.proposeParameters(self, model)
            samplingHelpers.computeAcceptanceProbability(self, stateEstimator, model)
            
            if (np.random.random(1) < self.acceptProb[iteration, :]):
                self.acceptParameters()
            else:
                self.rejectParameters()
            
            if self.settings['verbose']:
                print("Current unrestricted parameters: " + str(self.unrestrictedParameters[iteration, :]) + ".")
                print("Current restricted parameters: " + str(self.restrictedParameters[iteration, :]) + ".")
                input("Press ENTER to continue...")

            if np.remainder(iteration + 1, self.settings['nProgressReport']) == 0:
                helpers.printProgressReportToScreen(self)
            #self.printSimulationToFile()

        self.results = {}
        self.results.update({'parameterMeanEstimates': np.mean(self.restrictedParameters[noBurnInIters:noIters, :], axis=0)})
        self.results.update({'stateMeanEstimates': np.mean(self.states[noBurnInIters:noIters, :], axis=0)})
        self.results.update({'stateVarEstimates': np.var(self.states[noBurnInIters:noIters, :], axis=0)})
        self.results.update({'parameterTrace': self.restrictedParameters[noBurnInIters:noIters, :]})

    def plot(self):
        helpers.plotResults(self)

    def acceptParameters(self):
        iteration = self.currentIteration
        if self.proposedRestrictedParameters[iteration, 1] > 1.0:
            rasie
        self.restrictedParameters[iteration, :] = self.proposedRestrictedParameters[iteration, :]
        self.unrestrictedParameters[iteration, :] = self.proposedUnrestrictedParameters[iteration, :]
        self.logJacobian[iteration] = self.proposedLogJacobian[iteration]
        self.logPrior[iteration, :] = self.proposedLogPrior[iteration, :]
        self.logLikelihood[iteration, :] = self.proposedLogLikelihood[iteration, :]
        self.states[iteration, :] = self.proposedStates[iteration, :]
        self.gradient[iteration, :] = self.proposedGradient[iteration, :]
        self.hessian[iteration, :, :] = self.proposedHessian[iteration, :, :]
        self.accepted[iteration] = 1.0

    def rejectParameters(self):
        iteration = self.currentIteration
        self.restrictedParameters[iteration, :] = self.restrictedParameters[iteration - 1, :]
        self.unrestrictedParameters[iteration, :] = self.unrestrictedParameters[iteration - 1, :]
        self.logJacobian[iteration] = self.logJacobian[iteration - 1]
        self.logPrior[iteration, :] = self.logPrior[iteration - 1, :]
        self.logLikelihood[iteration, :] = self.logLikelihood[iteration - 1, :]
        self.states[iteration, :] = self.states[iteration - 1, :]
        self.gradient[iteration, :] = self.gradient[iteration - 1, :]
        self.hessian[iteration, :, :] = self.hessian[iteration - 1, :, :]
        self.accepted[iteration] = 0.0
