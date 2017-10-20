import numpy as np
import matplotlib.pylab as plt

from models import linearGaussianModelWithMean
from models.helpers import getInferenceModel
from state import kalmanMethods

def run():     
    # System model
    systemModel = linearGaussianModelWithMean.model()
    systemModel.parameters['mu'] = 0.2
    systemModel.parameters['phi'] = 0.5
    systemModel.parameters['sigma_v'] = 1.0
    systemModel.parameters['sigma_e'] = 0.01
    systemModel.noObservations = 5000
    systemModel.initialState = 0.0
    systemModel.generateData()

    # Inference model
    inferenceModel = getInferenceModel(systemModel, parametersToEstimate = ('mu', 'phi', 'sigma_v'))

    # Kalman filter and smoother
    kalman = kalmanMethods.FilteringSmoothing()
    kalman.settings['initialState'] = systemModel.initialState
    kalman.settings['initialCovariance'] = 1e-5
    kalman.settings['estimateGradients'] = True
    kalman.settings['estimateHessians'] = True

    kalman.smoother(inferenceModel)
    plt.subplot(311)
    plt.plot(np.arange(systemModel.noObservations), systemModel.states)
    plt.ylabel("states")
    plt.xlabel("time")
    plt.subplot(312)
    plt.plot(np.arange(systemModel.noObservations), kalman.filteredStateEstimate - systemModel.states[:, 0])
    plt.ylabel("error in filtered state estimate")
    plt.xlabel("time")
    plt.title('State estimation')
    plt.subplot(313)
    plt.plot(np.arange(systemModel.noObservations), kalman.smoothedStateEstimate[:, 0] - systemModel.states[:, 0])
    plt.ylabel("error in smoothed state estimate")
    plt.xlabel("time")
    plt.title('State estimation')    
    plt.show()

    print("MSE of filter: " + str(np.mean((kalman.filteredStateEstimate - systemModel.states[:, 0])**2)))
    print("MSE of smoother: " + str(np.mean((kalman.smoothedStateEstimate[:, 0] - systemModel.states[:, 0])**2)))


    # Mu
    inferenceModel = getInferenceModel(systemModel, parametersToEstimate = ('mu'))

    grid_mu = np.arange(-1, 1, 0.05)
    logLikelihood_mu = np.zeros(len(grid_mu))
    gradient_mu = np.zeros(len(grid_mu))
    natural_gradient_mu = np.zeros(len(grid_mu))

    for i in range(len(grid_mu)):
        inferenceModel.storeRestrictedParameters(grid_mu[i])
        kalman.smoother(inferenceModel)
        logLikelihood_mu[i] = kalman.logLikelihood
        gradient_mu[i] = kalman.gradientInternal
        natural_gradient_mu[i] = kalman.gradientInternal / kalman.hessianInternal

    # Phi
    inferenceModel = getInferenceModel(systemModel, parametersToEstimate = ('phi'))

    grid_phi = np.arange(-0.9, 1, 0.1)
    logLikelihood_phi = np.zeros(len(grid_phi))
    gradient_phi = np.zeros(len(grid_phi))
    natural_gradient_phi = np.zeros(len(grid_phi))

    for i in range(len(grid_phi)):
        inferenceModel.storeRestrictedParameters(grid_phi[i])
        kalman.smoother(inferenceModel)
        logLikelihood_phi[i] = kalman.logLikelihood
        gradient_phi[i] = kalman.gradientInternal
        natural_gradient_phi[i] = kalman.gradientInternal / kalman.hessianInternal

    # Sigma_v
    inferenceModel = getInferenceModel(systemModel, parametersToEstimate = ('sigma_v'))

    grid_sigmav = np.arange(0.5, 2, 0.1)
    logLikelihood_sigmav = np.zeros(len(grid_sigmav))
    gradient_sigmav = np.zeros(len(grid_sigmav))
    natural_gradient_sigmav = np.zeros(len(grid_sigmav))

    for i in range(len(grid_sigmav)):
        inferenceModel.storeUnrestrictedParameters(np.log(grid_sigmav[i]))
        kalman.smoother(inferenceModel)
        logLikelihood_sigmav[i] = kalman.logLikelihood
        gradient_sigmav[i] = kalman.gradientInternal
        natural_gradient_sigmav[i] = kalman.gradientInternal / kalman.hessianInternal


    #Plotting
    plt.figure()
    plt.subplot(331)
    plt.plot(grid_mu, gradient_mu)
    plt.xlabel("mu")
    plt.ylabel("Gradient of mu")
    plt.axvline(x=systemModel.parameters['mu'], color='r')
    plt.axhline(y=0.0, color='r')
    plt.subplot(332)
    plt.plot(grid_mu, natural_gradient_mu)
    plt.xlabel("mu")
    plt.ylabel("Natural gradient of mu")
    plt.axvline(x=systemModel.parameters['mu'], color='r')
    plt.axhline(y=0.0, color='r')    
    plt.subplot(333)
    plt.plot(grid_mu, logLikelihood_mu)
    plt.xlabel("mu")
    plt.ylabel("Likelihood")
    plt.axvline(x=systemModel.parameters['mu'], color='r')

    plt.subplot(334)
    plt.plot(grid_phi, gradient_phi)
    plt.xlabel("phi")
    plt.ylabel("Gradient of phi")
    plt.axvline(x=systemModel.parameters['phi'], color='r')
    plt.axhline(y=0.0, color='r')
    plt.subplot(335)
    plt.plot(grid_phi, natural_gradient_phi)
    plt.xlabel("phi")
    plt.ylabel("Natural gradient of phi")
    plt.axvline(x=systemModel.parameters['phi'], color='r')
    plt.axhline(y=0.0, color='r')    
    plt.subplot(336)
    plt.plot(grid_phi, logLikelihood_phi)
    plt.xlabel("phi")
    plt.ylabel("Likelihood")
    plt.axvline(x=systemModel.parameters['phi'], color='r')

    plt.subplot(337)
    plt.plot(grid_sigmav, gradient_sigmav)
    plt.xlabel("sigma_v")
    plt.ylabel("Gradient of sigma_v")
    plt.axvline(x=systemModel.parameters['sigma_v'], color='r')
    plt.axhline(y=0.0, color='r')
    plt.subplot(338)
    plt.plot(grid_sigmav, natural_gradient_sigmav)
    plt.xlabel("sigma_v")
    plt.ylabel("Natural gradient of sigma_v")
    plt.axvline(x=systemModel.parameters['sigma_v'], color='r')
    plt.axhline(y=0.0, color='r')    
    plt.subplot(339)
    plt.plot(grid_sigmav, logLikelihood_sigmav)
    plt.xlabel("sigma_v")
    plt.ylabel("Likelihood")
    plt.axvline(x=systemModel.parameters['sigma_v'], color='r')        
    plt.show()

    # grid_mu = np.arange(-1, 1, 0.01)
    # gradient_mu = np.zeros(len(grid_mu))
    # for i in range(len(grid_mu)):
    #     inferenceModel.parameters['mu'] = grid_mu[i]
    #     kalman.smoother(inferenceModel)
    #     gradient_mu[i] = kalman.gradient['mu']

    # plt.plot(grid_mu, gradient_mu)
    # plt.show()

