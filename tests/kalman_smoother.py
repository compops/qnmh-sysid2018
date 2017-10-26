import numpy as np
import matplotlib.pylab as plt

from models.linear_gaussian_model import SystemModel
from state.kalman_methods.main import KalmanMethods

def run():
    # System model
    sys_model = SystemModel()
    sys_model.params['mu'] = 0.20
    sys_model.params['phi'] = 0.50
    sys_model.params['sigma_v'] = 1.00
    sys_model.params['sigma_e'] = 0.10
    sys_model.no_obs = 1000
    sys_model.initial_state = 0.0

    #sys_model.generate_data(file_name="data/linear_gaussian_model/linear_gaussian_model_T1000_goodSNR.csv")
    sys_model.import_data(file_name="data/linear_gaussian_model/linear_gaussian_model_T1000_goodSNR.csv")

    # Inference model
    sys_model.fix_true_params()
    sys_model.create_inference_model(params_to_estimate = ('mu', 'phi', 'sigma_v'))

    # Kalman filter and smoother
    kalman_settings = {'initial_state': sys_model.initial_state,
                    'initial_cov': 1e-5,
                    'estimate_gradient': True,
                    'estimate_hessian': True
                    }
    kf = KalmanMethods(kalman_settings)

    kf.smoother(sys_model)

    plt.subplot(311)
    plt.plot(np.arange(sys_model.no_obs+1), sys_model.states)
    plt.ylabel("states")
    plt.xlabel("time")
    plt.subplot(312)
    plt.plot(np.arange(sys_model.no_obs+1), kf.filt_state_est - sys_model.states[:, 0])
    plt.ylabel("error in filtered state estimate")
    plt.xlabel("time")
    plt.title('State estimation')
    plt.subplot(313)
    plt.plot(np.arange(sys_model.no_obs+1), kf.smo_state_est[:, 0] - sys_model.states[:, 0])
    plt.ylabel("error in smoothed state estimate")
    plt.xlabel("time")
    plt.title('State estimation')
    plt.show()

    print("MSE of filter: " + str(np.mean((kf.filt_state_est - sys_model.states[:, 0])**2)))
    print("MSE of smoother: " + str(np.mean((kf.smo_state_est[:, 0] - sys_model.states[:, 0])**2)))

    # Mu
    sys_model.create_inference_model(params_to_estimate = ('mu'))

    grid_mu = np.arange(-0.95, 0.90, 0.05)
    log_like_mu = np.zeros(len(grid_mu))
    gradient_mu = np.zeros(len(grid_mu))
    natural_gradient_mu = np.zeros(len(grid_mu))

    for i in range(len(grid_mu)):
        sys_model.store_params(grid_mu[i])
        log_like_mu[i] = kf.log_like
        kf.smoother(sys_model)
        gradient_mu[i] = kf.gradient_internal
        natural_gradient_mu[i] = kf.gradient_internal / kf.hessian_internal
        gradient_mu[i] /= (1.0 - sys_model.params['phi'])

    # Phi
    sys_model.create_inference_model(params_to_estimate = ('phi'))

    grid_phi = np.arange(-0.9, 1, 0.1)
    log_like_phi = np.zeros(len(grid_phi))
    gradient_phi = np.zeros(len(grid_phi))
    natural_gradient_phi = np.zeros(len(grid_phi))

    for i in range(len(grid_phi)):
        sys_model.store_params(grid_phi[i])
        kf.smoother(sys_model)
        log_like_phi[i] = kf.log_like
        gradient_phi[i] = kf.gradient_internal
        natural_gradient_phi[i] = kf.gradient_internal / kf.hessian_internal
        gradient_phi[i] /= (1.0 - sys_model.params['phi']**2)

    # Sigma_v
    sys_model.create_inference_model(params_to_estimate = ('sigma_v'))

    grid_sigmav = np.arange(0.5, 2, 0.1)
    log_like_sigmav = np.zeros(len(grid_sigmav))
    gradient_sigmav = np.zeros(len(grid_sigmav))
    natural_gradient_sigmav = np.zeros(len(grid_sigmav))

    for i in range(len(grid_sigmav)):
        sys_model.store_params(grid_sigmav[i])
        kf.smoother(sys_model)
        log_like_sigmav[i] = kf.log_like
        gradient_sigmav[i] = kf.gradient_internal
        natural_gradient_sigmav[i] = kf.gradient_internal / kf.hessian_internal
        gradient_sigmav[i] /= grid_sigmav[i]


    #Plotting
    plt.figure()
    plt.subplot(331)
    plt.plot(grid_mu, gradient_mu)
    plt.xlabel("mu")
    plt.ylabel("Gradient of mu")
    plt.axvline(x=sys_model.true_params['mu'], color='r')
    plt.axhline(y=0.0, color='r')
    plt.subplot(332)
    plt.plot(grid_mu, natural_gradient_mu)
    plt.xlabel("mu")
    plt.ylabel("Natural gradient of mu")
    plt.axvline(x=sys_model.true_params['mu'], color='r')
    plt.axhline(y=0.0, color='r')
    plt.subplot(333)
    plt.plot(grid_mu, log_like_mu)
    plt.xlabel("mu")
    plt.ylabel("Likelihood")
    plt.axvline(x=sys_model.true_params['mu'], color='r')

    plt.subplot(334)
    plt.plot(grid_phi, gradient_phi)
    plt.xlabel("phi")
    plt.ylabel("Gradient of phi")
    plt.axvline(x=sys_model.true_params['phi'], color='r')
    plt.axhline(y=0.0, color='r')
    plt.subplot(335)
    plt.plot(grid_phi, natural_gradient_phi)
    plt.xlabel("phi")
    plt.ylabel("Natural gradient of phi")
    plt.axvline(x=sys_model.true_params['phi'], color='r')
    plt.axhline(y=0.0, color='r')
    plt.subplot(336)
    plt.plot(grid_phi, log_like_phi)
    plt.xlabel("phi")
    plt.ylabel("Likelihood")
    plt.axvline(x=sys_model.true_params['phi'], color='r')

    plt.subplot(337)
    plt.plot(grid_sigmav, gradient_sigmav)
    plt.xlabel("sigma_v")
    plt.ylabel("Gradient of sigma_v")
    plt.axvline(x=sys_model.true_params['sigma_v'], color='r')
    plt.axhline(y=0.0, color='r')
    plt.subplot(338)
    plt.plot(grid_sigmav, natural_gradient_sigmav)
    plt.xlabel("sigma_v")
    plt.ylabel("Natural gradient of sigma_v")
    plt.axvline(x=sys_model.true_params['sigma_v'], color='r')
    plt.axhline(y=0.0, color='r')
    plt.subplot(339)
    plt.plot(grid_sigmav, log_like_sigmav)
    plt.xlabel("sigma_v")
    plt.ylabel("Likelihood")
    plt.axvline(x=sys_model.true_params['sigma_v'], color='r')
    plt.show()

    # grid_mu = np.arange(-1, 1, 0.01)
    # gradient_mu = np.zeros(len(grid_mu))
    # for i in range(len(grid_mu)):
    #     inferenceModel.params['mu'] = grid_mu[i]
    #     kalman.smoother(inferenceModel)
    #     gradient_mu[i] = kalman.gradient['mu']

    # plt.plot(grid_mu, gradient_mu)
    # plt.show()

