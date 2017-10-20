import numpy as np
import matplotlib.pylab as plt

from models import linearGaussianModelWithMean
from models.helpers import getInferenceModel
from state import kalmanMethods
from parameter.mcmc import metropolisHastings

def run():
    # System model
    systemModel = linearGaussianModelWithMean.model()
    systemModel.parameters['mu'] = 0.20
    systemModel.parameters['phi'] = 0.80
    systemModel.parameters['sigma_v'] = 1.00
    systemModel.parameters['sigma_e'] = 0.10
    systemModel.noObservations = 1000
    systemModel.initialState = 0.0
    systemModel.generateData()

    # Inference model
    inferenceModel = getInferenceModel(systemModel, 
                                       parametersToEstimate = ('mu', 'phi', 'sigma_v'),
                                       unRestrictedParameters = True)

    # Kalman filter
    kalman = kalmanMethods.FilteringSmoothing()
    kalmanSettings = {'initialState': systemModel.initialState,
                      'initialCovariance': 1e-5,
                      'estimateGradients': True,
                      'estimateHessians': True
    }
    kalman.settings = kalmanSettings
    
    # Metropolis-Hastings
    mhSettings = {'noIters': 1000, 
                  'noBurnInIters': 200, 
                  'stepSize': 1.0, 
                  'initialParameters': (0.2, 0.5, 1.0), 
                  'verbose': False,
                  'hessianEstimate': 'kalman',
                  'SR1UpdateLimit': '1e-8',
                  'printWarningsForUnstableSystems': True,
                  'memoryLength': 20,
                  'initialHessian': 1e-4,
                  'trustRegionSize': None
                  }
    mhSampler = metropolisHastings.ParameterEstimator(mhSettings)
    mhSampler.run(kalman, inferenceModel, 'mh2')
    mhSampler.plot()

    return(mhSampler)

