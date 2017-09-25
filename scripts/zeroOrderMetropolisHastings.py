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
    kalman.settings['initialState'] = systemModel.initialState
    kalman.settings['initialCovariance'] = 1e-5

    # Metropolis-Hastings
    stepSize = 2.562 / np.sqrt(3)
    posteriorCovariance = np.array((0.00013185, 0.00071256, 0.0006298,
                                    0.00071256, 0.0049845,  0.00460153,
                                    0.0006298,  0.00460153, 0.00439571)).reshape((3,3))
    settings = {'noIters': 1000, 
                'noBurnInIters': 100, 
                'stepSize': stepSize, 
                'initialParameters': (0.0, 0.0, 0.5), 
                'hessianEstimate': posteriorCovariance,
                'verbose': False,
                'printWarningsForUnstableSystems': False
                }
    mhSampler = metropolisHastings.ParameterEstimator(settings)
    mhSampler.run(kalman, inferenceModel, 'mh0')
    mhSampler.plot()

