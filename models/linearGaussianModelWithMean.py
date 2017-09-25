import numpy as np
from scipy.stats import norm

from models.helpers import *
from models.distributions import *

class model(object):
    modelName = "Linear Gaussian SSM with four parameters"
    modelType = "Data generation model"
    parameterisation = "standard"
    filePrefix = "lgss"
    states = []
    observations = []
    initialState = []
    parameters = {}
    noParameters = 4
    prior = {'mu': (0.0, 0.2), 'phi': (0.9, 0.05), 'sigma_v': (0.2, 0.2), 'sigma_e': (2.0, 2.0)}

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
        gradient.update({'phi': normalLogPDFgradient(self.parameters['phi'], self.prior['phi'][0], self.prior['phi'][1])})
        gradient.update({'sigma_v': gammaLogPDFgradient(self.parameters['sigma_v'], a=self.prior['sigma_v'][0], b=self.prior['sigma_v'][1])})
        gradient.update({'sigma_e': gammaLogPDFgradient(self.parameters['sigma_e'], a=self.prior['sigma_e'][0], b=self.prior['sigma_e'][1])})
        return(gradient)

    def hessianLogPrior(self):
        hessian = {}
        hessian.update({'mu': normalLogPDFhessian(self.parameters['mu'], self.prior['mu'][0], self.prior['mu'][1])})
        hessian.update({'phi': normalLogPDFhessian(self.parameters['phi'], self.prior['phi'][0], self.prior['phi'][1])})
        hessian.update({'sigma_v': gammaLogPDFhessian(self.parameters['sigma_v'], a=self.prior['sigma_v'][0], b=self.prior['sigma_v'][1])})
        hessian.update({'sigma_e': gammaLogPDFhessian(self.parameters['sigma_e'], a=self.prior['sigma_e'][0], b=self.prior['sigma_e'][1])})
        return(hessian)

    def gradientLogJointLikelihood(self, nextState, currentState, currentObservation):
        px = nextState - self.parameters['mu'] - self.parameters['phi'] * (currentState - self.parameters['mu'])
        py = currentObservation - currentState
        gradient = {}
        gradient.update({'mu': (1.0 - self.parameters['phi']) * self.parameters['sigma_v']**(-2) * px})
        gradient.update({'mu': (currentState - self.parameters['mu']) * self.parameters['sigma_v']**(-2) * px})
        gradient.update({'mu': self.parameters['sigma_v']**(-3) * px**2 - self.parameters['sigma_v']**(-1)})
        gradient.update({'mu': self.parameters['sigma_e']**(-3) * py**2 - self.parameters['sigma_e']**(-1)   })
        return(gradient)         

    # Define standard methods for the model struct
    storeParameters = template_storeParameters

    generateData = template_generateData
    importData = template_importData
