import numpy as np
import matplotlib.pylab as plt

from models import linearGaussianModelWithMean
from models.helpers import getInferenceModel
from state import kalmanMethods

def run():     
    # System model
    systemModel = linearGaussianModelWithMean.model()
    systemModel.parameters['mu'] = 0.2
    systemModel.parameters['phi'] = 0.75
    systemModel.parameters['sigma_v'] = 0.16
    systemModel.parameters['sigma_e'] = 0.1
    systemModel.noObservations = 500
    systemModel.initialState = 0.0
    systemModel.generateData()

    # Inference model
    inferenceModel = getInferenceModel(systemModel, parametersToEstimate = ('mu', 'phi'))

    # Kalman filter and smoother
    kalman = kalmanMethods.FilteringSmoothing()
    kalman.settings['initialState'] = systemModel.initialState
    kalman.settings['initialCovariance'] = 1e-5
    kalman.settings['estimateGradients'] = True
    kalman.settings['estimateHessians'] = False

    kalman.smoother(inferenceModel)
    #plt.plot(np.arange(systemModel.noObservations), systemModel.states, np.arange(systemModel.noObservations), kalman.filteredStateEstimate)
    #plt.show()

    grid_phi = np.arange(-1, 1, 0.01)
    logLikelihood_phi = np.zeros(len(grid_phi))
    gradient_phi = np.zeros(len(grid_phi))
    for i in range(len(grid_phi)):
        inferenceModel.parameters['phi'] = grid_phi[i]
        kalman.smoother(inferenceModel)
        logLikelihood_phi[i] = kalman.logLikelihood
        gradient_phi[i] = kalman.gradient['phi']

    plt.figure()
    plt.subplot(211)
    plt.plot(grid_phi, gradient_phi)
    plt.subplot(212)
    plt.plot(grid_phi, logLikelihood_phi)
    plt.show()

    # grid_mu = np.arange(-1, 1, 0.01)
    # gradient_mu = np.zeros(len(grid_mu))
    # for i in range(len(grid_mu)):
    #     inferenceModel.parameters['mu'] = grid_mu[i]
    #     kalman.smoother(inferenceModel)
    #     gradient_mu[i] = kalman.gradient['mu']

    # plt.plot(grid_mu, gradient_mu)
    # plt.show()

