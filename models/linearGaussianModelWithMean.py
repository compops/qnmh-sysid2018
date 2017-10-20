import numpy as np
from scipy.stats import norm

from models.helpers import *
from models.distributions import *

class model(object):
    modelName = "Linear Gaussian SSM with four parameters"
    modelType = "Data generation model"
    parameterisation = "standard"
    currentMode = "Parameters in restricted space"
    filePrefix = "lgss"
    states = []
    observations = []
    initialState = []
    parameters = {}
    unrestrictedParameters = {}
    noParameters = 4
    prior = {'mu': (0.0, 1.0), 'phi': (0.5, 1.0), 'sigma_v': (2.0, 2.0), 'sigma_e': (2.0, 2.0)}

    def generateInitialState(self, noSamples):
        return self.parameters['mu'] + np.random.normal(size=(1, noSamples)) * self.parameters['sigma_v'] / np.sqrt(1.0 - self.parameters['phi']**2)

    def generateState(self, currentState):
        return self.parameters['mu'] + self.parameters['phi'] * (currentState - self.parameters['mu']) + self.parameters['sigma_v'] * np.random.randn(1, len(currentState))

    def evaluateState(self, nextState, currentState):
        return norm.pdf(nextState, self.parameters['mu'] + self.parameters['phi'] * (currentState - self.parameters['mu']), self.parameters['sigma_v'])

    def generateObservation(self, currentState):
        return currentState + self.parameters['sigma_e'] * np.random.randn(1, len(currentState))

    def evaluateObservation(self, currentState, currentObservation):
        return norm.logpdf(currentObservation, currentState, self.parameters['sigma_e'])
    
    def areParametersValid(self):
        out = True
        if(np.abs(self.parameters['phi']) > 1.0):
            out = False
        if(self.parameters['sigma_v'] < 0.0):
            out = False
        if(self.parameters['sigma_e'] < 0.0):
            out = False
        return(out)
    
    def logPrior(self):
        prior = {}
        prior.update({'mu': normalLogPDF(self.parameters['mu'], self.prior['mu'][0], self.prior['mu'][1])})
        prior.update({'phi': normalLogPDF(self.parameters['phi'], self.prior['phi'][0], self.prior['phi'][1])})
        prior.update({'sigma_v': gammaLogPDF(self.parameters['sigma_v'], a=self.prior['sigma_v'][0], b=self.prior['sigma_v'][1])})
        prior.update({'sigma_e': gammaLogPDF(self.parameters['sigma_e'], a=self.prior['sigma_e'][0], b=self.prior['sigma_e'][1])})
        priorVector = [prior[item] for item in prior]

        if self.modelType is "Inference model":
            priorVector = priorVector[self.parametersToEstimateIndex]
        return(prior, np.sum(priorVector))
    
    def gradientLogPrior(self):
        gradient = {}
        gradient.update({'mu': normalLogPDFgradient(self.parameters['mu'], self.prior['mu'][0], self.prior['mu'][1])})
        gradient.update({'phi': (1.0 - self.parameters['phi']**2) * normalLogPDFgradient(self.parameters['phi'], self.prior['phi'][0], self.prior['phi'][1])})
        gradient.update({'sigma_v': self.parameters['sigma_v'] * gammaLogPDFgradient(self.parameters['sigma_v'], a=self.prior['sigma_v'][0], b=self.prior['sigma_v'][1])})
        gradient.update({'sigma_e': self.parameters['sigma_e'] * gammaLogPDFgradient(self.parameters['sigma_e'], a=self.prior['sigma_e'][0], b=self.prior['sigma_e'][1])})
        return(gradient)

    def hessianLogPrior(self):
        hessian = {}
        phi = (1.0 - self.parameters['phi']**2) * normalLogPDFhessian(self.parameters['phi'], self.prior['phi'][0], self.prior['phi'][1])
        phi += (1.0 - 2.0 * self.parameters['phi']**2) * normalLogPDFgradient(self.parameters['mu'], self.prior['mu'][0], self.prior['mu'][1])
        sigmav = gammaLogPDFhessian(self.parameters['sigma_v'], a=self.prior['sigma_v'][0], b=self.prior['sigma_v'][1]) * self.parameters['sigma_v']
        sigmav += gammaLogPDFgradient(self.parameters['sigma_v'], a=self.prior['sigma_v'][0], b=self.prior['sigma_v'][1]) * self.parameters['sigma_v']
        hessian.update({'mu': normalLogPDFhessian(self.parameters['mu'], self.prior['mu'][0], self.prior['mu'][1])})
        hessian.update({'phi': phi})
        hessian.update({'sigma_v': sigmav})
        hessian.update({'sigma_e': gammaLogPDFhessian(self.parameters['sigma_e'], a=self.prior['sigma_e'][0], b=self.prior['sigma_e'][1])})
        return(hessian)

    def gradientLogJointLikelihood(self, nextState, currentState, currentObservation):
        px = nextState - self.parameters['mu'] - self.parameters['phi'] * (currentState - self.parameters['mu'])
        py = currentObservation - currentState
        Q = self.parameters['sigma_v']**(-2)
        R = self.parameters['sigma_e']**(-2)

        gradient = {}
        gradient.update({'mu': (1.0 - self.parameters['phi']) * Q * px})
        gradient.update({'phi': (currentState - self.parameters['mu']) * Q * px * (1.0 - self.parameters['phi']**2)})
        gradient.update({'sigma_v': Q * px**2 - 1.0})
        gradient.update({'sigma_e': R * py**2 - 1.0})
        return(gradient)         

    def transformParametersToUnrestricted(self):
        #print("transformParametersToUnrestricted, self.parameters: " + str(self.parameters))
        try:
            self.unrestrictedParameters['mu'] = self.parameters['mu']
            self.unrestrictedParameters['phi'] = np.arctanh(self.parameters['phi'])
            self.unrestrictedParameters['sigma_v'] = np.log(self.parameters['sigma_v'])
            self.unrestrictedParameters['sigma_e'] = np.log(self.parameters['sigma_e'])        
        except:
            print(self.parameters)
            input("ENTER")
        #print("transformParametersToUnrestricted, self.unrestrictedParameters: " + str(self.unrestrictedParameters))
    
    def transformParametersFromUnrestricted(self):
        #print("transformParametersFromUnrestricted, self.unrestrictedParameters: " + str(self.unrestrictedParameters))
        self.parameters['mu'] = self.unrestrictedParameters['mu']
        self.parameters['phi'] = np.tanh(self.unrestrictedParameters['phi'])
        self.parameters['sigma_v'] = np.exp(self.unrestrictedParameters['sigma_v'])
        self.parameters['sigma_e'] = np.exp(self.unrestrictedParameters['sigma_e'])
        #print("transformParametersFromUnrestricted, self.parameters: " + str(self.parameters))
    
    def logJacobian(self):
        jacobian = {}
        jacobian.update({'mu': 0.0})
        jacobian.update({'phi': np.log(1.0 - self.parameters['phi']**2)})
        jacobian.update({'sigma_v': np.log(self.parameters['sigma_v'])})
        jacobian.update({'sigma_e': np.log(self.parameters['sigma_e'])})
        output = 0.0
        if self.noParametersToEstimate > 1:
            for param in self.parametersToEstimate:
                output += jacobian[param]
        else:
            output += jacobian[self.parametersToEstimate]
        
        return(output)
    
    # Define standard methods for the model struct
    generateData = template_generateData
    importData = template_importData
    storeUnrestrictedParameters = template_storeUnrestrictedParameters
    storeRestrictedParameters = template_storeRestrictedParameters
    getUnrestrictedParameters = template_getUnrestrictedParameters
    getRestrictedParameters = template_getRestrictedParameters