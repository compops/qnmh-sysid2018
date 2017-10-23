import pickle
import uuid
import numpy as np
import matplotlib.pylab as plt

from models import linearGaussianModelWithMean
from models.helpers import getInferenceModel
from state import kalmanMethods
from parameter.mcmc import metropolisHastings

def run():
    # Set random seed
    np.random.seed(234878)

    # System model
    systemModel = linearGaussianModelWithMean.model()
    systemModel.parameters['mu'] = 0.20
    systemModel.parameters['phi'] = 0.80
    systemModel.parameters['sigma_v'] = 1.00
    systemModel.parameters['sigma_e'] = 0.10
    systemModel.noObservations = 1000
    systemModel.initialState = 0.0
    #systemModel.generateData(fileName="data/lgss/lgssT1000_smallR.csv")
    systemModel.importData(fileName="data/lgss/lgssT1000_smallR.csv")

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
                  'noBurnInIters': 500, 
                  'stepSize': 0.5, 
                  'initialParameters': (0.2, 0.5, 1.0), 
                  'verbose': False,
                  'waitForENTER': False,
                  'informOfHessianCorrection': True,
                  'hessianEstimate': 'SR1',
                  'SR1UpdateLimit': 1e-10,
                  'printWarningsForUnstableSystems': True,
                  'memoryLength': 20,
                  'initialHessian': 1e-2,
                  'trustRegionSize': None,
                  'hessianCorrectionApproach': 'replace',
                  'hessianEstimateOnlyAcceptedInformation': True
                  }
    mhSampler = metropolisHastings.ParameterEstimator(mhSettings)
    mhSampler.run(kalman, inferenceModel, 'mh2')
    mhSampler.plot()

    # Save run to file
    filename = str(uuid.uuid4())
    with open("runs/" + filename + ".pickle", 'wb') as f:
        pickle.dump(mhSampler, f)
    print("Saved run to " + filename + ".")
