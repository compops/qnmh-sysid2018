import numpy as np
import matplotlib.pylab as plt

from models.linear_gaussian_model import SystemModel
from state.particle_methods.main import ParticleMethods

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
    particle_settings = {'resampling_method': 'multinomial',
                         'no_particles': 1000,
                         'estimate_gradient': True,
                         'fixed_lag': 5,
                         'generate_initial_state': True,
                         'initial_state': 0.0
                        }
    pf = ParticleMethods(particle_settings)
    pf.smoother_linear_gaussian_model(sys_model)

    plt.subplot(311)
    plt.plot(np.arange(sys_model.no_obs+1), sys_model.states)
    plt.ylabel("states")
    plt.xlabel("time")
    plt.subplot(312)
    plt.plot(np.arange(sys_model.no_obs+1), pf.filt_state_est[:, 0] - sys_model.states[:, 0])
    plt.ylabel("error in filtered state estimate")
    plt.xlabel("time")
    plt.title('State estimation')
    plt.subplot(313)
    plt.plot(np.arange(sys_model.no_obs+1), pf.smo_state_est[:, 0] - sys_model.states[:, 0])
    plt.ylabel("error in smoothed state estimate")
    plt.xlabel("time")
    plt.title('State estimation')
    plt.show()

    print("MSE of filter: " + str(np.mean((pf.filt_state_est - sys_model.states[:, 0])**2)))
    print("MSE of smoother: " + str(np.mean((pf.smo_state_est[:, 0] - sys_model.states[:, 0])**2)))

    # Mu
    sys_model.create_inference_model(params_to_estimate = ('mu'))

    grid_mu = np.arange(-1, 1, 0.05)
    log_like_mu = np.zeros(len(grid_mu))
    gradient_mu = np.zeros(len(grid_mu))

    for i in range(len(grid_mu)):
        sys_model.store_params(grid_mu[i])
        log_like_mu[i] = pf.log_like
        pf.smoother_linear_gaussian_model(sys_model)
        gradient_mu[i] = pf.gradient_internal

    # Phi
    sys_model.create_inference_model(params_to_estimate = ('phi'))

    grid_phi = np.arange(-0.9, 1, 0.1)
    log_like_phi = np.zeros(len(grid_phi))
    gradient_phi = np.zeros(len(grid_phi))

    for i in range(len(grid_phi)):
        sys_model.store_params(grid_phi[i])
        pf.smoother_linear_gaussian_model(sys_model)
        log_like_phi[i] = pf.log_like
        gradient_phi[i] = pf.gradient_internal

    # Sigma_v
    sys_model.create_inference_model(params_to_estimate = ('sigma_v'))

    grid_sigmav = np.arange(0.5, 2, 0.1)
    log_like_sigmav = np.zeros(len(grid_sigmav))
    gradient_sigmav = np.zeros(len(grid_sigmav))

    for i in range(len(grid_sigmav)):
        sys_model.store_params(np.log(grid_sigmav[i]))
        pf.smoother_linear_gaussian_model(sys_model)
        log_like_sigmav[i] = pf.log_like
        gradient_sigmav[i] = pf.gradient_internal


    #Plotting
    plt.figure()
    plt.subplot(321)
    plt.plot(grid_mu, gradient_mu)
    plt.xlabel("mu")
    plt.ylabel("Gradient of mu")
    plt.axvline(x=sys_model.true_params['mu'], color='r')
    plt.axhline(y=0.0, color='r')
    plt.subplot(322)
    plt.plot(grid_mu, log_like_mu)
    plt.xlabel("mu")
    plt.ylabel("Likelihood")
    plt.axvline(x=sys_model.true_params['mu'], color='r')

    plt.subplot(323)
    plt.plot(grid_phi, gradient_phi)
    plt.xlabel("phi")
    plt.ylabel("Gradient of phi")
    plt.axvline(x=sys_model.true_params['phi'], color='r')
    plt.axhline(y=0.0, color='r')
    plt.subplot(324)
    plt.plot(grid_phi, log_like_phi)
    plt.xlabel("phi")
    plt.ylabel("Likelihood")
    plt.axvline(x=sys_model.true_params['phi'], color='r')

    plt.subplot(325)
    plt.plot(grid_sigmav, gradient_sigmav)
    plt.xlabel("sigma_v")
    plt.ylabel("Gradient of sigma_v")
    plt.axvline(x=sys_model.true_params['sigma_v'], color='r')
    plt.axhline(y=0.0, color='r')
    plt.subplot(326)
    plt.plot(grid_sigmav, log_like_sigmav)
    plt.xlabel("sigma_v")
    plt.ylabel("Likelihood")
    plt.axvline(x=sys_model.true_params['sigma_v'], color='r')
    plt.show()