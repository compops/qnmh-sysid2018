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
    stepSize = 1.125 / np.sqrt(3.0**(1.0 / 3.0)) / systemModel.noObservations
    posteriorCovariance = np.array(( 0.00896405, -0.00202991, -0.08106153,
                                    -0.00202991,  0.0025671,   0.04450741,
                                    -0.08106153,  0.04450741,  1.44413436)).reshape((3,3))
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
    #mhSampler.plot()
