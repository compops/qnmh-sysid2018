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
    inferenceModel = getInferenceModel(systemModel, parametersToEstimate = ('mu', 'phi', 'sigma_v'))

    # Kalman filter
    kalman = kalmanMethods.FilteringSmoothing()
    kalmanSettings = {'initialState': systemModel.initialState,
                      'initialCovariance': 1e-5,
                      'estimateGradients': True,
                      'estimateHessians': False
    }
    kalman.settings = kalmanSettings
    
    # Metropolis-Hastings
    stepSize = 0.01 * 1.125 / np.sqrt(3.0**(1.0 / 3.0))
    posteriorCovariance = np.array((0.00013185, 0.00071256, 0.0006298,
                                    0.00071256, 0.0049845,  0.00460153,
                                    0.0006298,  0.00460153, 0.00439571)).reshape((3,3))
    mhSettings = {'noIters': 1000, 
                  'noBurnInIters': 200, 
                  'stepSize': stepSize, 
                  'initialParameters': (0.0, 0.0, 0.5), 
                  'hessianEstimate': posteriorCovariance,
                  'verbose': False,
                  'printWarningsForUnstableSystems': False
                  }
    mhSampler = metropolisHastings.ParameterEstimator(mhSettings)
    mhSampler.run(kalman, inferenceModel, 'mh1')
    mhSampler.plot()

