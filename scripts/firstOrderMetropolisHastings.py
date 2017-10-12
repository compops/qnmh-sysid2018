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
    systemModel.parameters['sigma_v'] = 1.0
    systemModel.parameters['sigma_e'] = 0.10
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
                      'estimateHessians': False
    }
    kalman.settings = kalmanSettings
    
    # Metropolis-Hastings
    stepSize = 0.015 * 1.125 / np.sqrt(len(inferenceModel.parametersToEstimate)**(1.0 / 3.0))
    posteriorCovariance = np.array((  0.02323771, -0.0047647,   0.02189193,
                                     -0.0047647,   0.00317307, -0.0088695,
                                      0.02189193, -0.0088695,   0.05392023)).reshape((3,3))
    mhSettings = {'noIters': 5000, 
                  'noBurnInIters': 1000, 
                  'stepSize': stepSize, 
                  'initialParameters': (0.5, 0.5, 0.5), 
                  'hessianEstimate': posteriorCovariance,
                  'verbose': False,
                  'printWarningsForUnstableSystems': False
                  }
    mhSampler = metropolisHastings.ParameterEstimator(mhSettings)
    mhSampler.run(kalman, inferenceModel, 'mh1')
    mhSampler.plot()

