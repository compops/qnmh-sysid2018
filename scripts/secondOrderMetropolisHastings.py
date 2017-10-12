import numpy as np
import matplotlib.pylab as plt

from models import linearGaussianModelWithMean
from models.helpers import getInferenceModel
from state import kalmanMethods
from parameter.mcmc import metropolisHastings

def run():
    # System model
    systemModel = linearGaussianModelWithMean.model()
    systemModel.parameters['mu'] = 0.2
    systemModel.parameters['phi'] = 0.8
    systemModel.parameters['sigma_v'] = 1.0
    systemModel.parameters['sigma_e'] = 0.1
    systemModel.noObservations = 500
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
    mhSettings = {'noIters': 5000, 
                  'noBurnInIters': 1000, 
                  'stepSize': 1.0, 
                  'initialParameters': (0.2, 0.8, 1.0), 
                  'verbose': False,
                  'printWarningsForUnstableSystems': False
                  }
    mhSampler = metropolisHastings.ParameterEstimator(mhSettings)
    mhSampler.run(kalman, inferenceModel, 'mh2')
    mhSampler.plot()

    return mhSampler

