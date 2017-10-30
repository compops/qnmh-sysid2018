import numpy as np
import matplotlib.pylab as plt

from models.phillips_curve_model import SystemModel
from state.particle_methods.main import ParticleMethods

def run(cython_code=False):
    # System model
    sys_model = SystemModel()
    sys_model.params['alpha'] = 0.45
    sys_model.params['phi'] = 0.76
    sys_model.params['beta'] = 0.02
    sys_model.params['sigma_e'] = 0.275
    sys_model.no_obs = 347
    sys_model.initial_state = 0.0

    sys_model.import_data(file_name="data/phillips_curve_model/sweden_1987_2015.csv")

    # Inference model
    sys_model.fix_true_params()
    sys_model.create_inference_model(params_to_estimate = ('alpha', 'phi', 'beta', 'sigma_e'))

    # Particle filter and smoother
    particle_settings = {'resampling_method': 'systematic',
                         'no_particles': 1000,
                         'estimate_gradient': True,
                         'estimate_hessian_segalweinstein': True,
                         'fixed_lag': 10,
                         'generate_initial_state': True
                        }
    pf = ParticleMethods(particle_settings)
    pf.smoother(sys_model)

    plt.subplot(211)
    plt.plot(np.arange(sys_model.no_obs+1), pf.filt_state_est[:, 0])
    plt.ylabel("filtered state estimate")
    plt.xlabel("time")
    plt.title('State estimation')
    plt.subplot(212)
    plt.plot(np.arange(sys_model.no_obs+1), pf.smo_state_est[:, 0])
    plt.ylabel("smoothed state estimate")
    plt.xlabel("time")
    plt.title('State estimation')
    plt.show()

    # Alpha
    sys_model.create_inference_model(params_to_estimate = ('alpha'))

    grid_alpha = np.arange(0.0, 1.0, 0.05)
    log_like_alpha = np.zeros(len(grid_alpha))
    gradient_alpha = np.zeros(len(grid_alpha))
    nat_gradient_alpha = np.zeros(len(grid_alpha))

    for i in range(len(grid_alpha)):
        sys_model.store_params(grid_alpha[i])
        log_like_alpha[i] = pf.log_like
        pf.smoother(sys_model)
        gradient_alpha[i] = pf.gradient_internal
        nat_gradient_alpha[i] = pf.gradient_internal / pf.hessian_internal

    # Phi
    sys_model.create_inference_model(params_to_estimate = ('phi'))

    grid_phi = np.arange(0.5, 1.0, 0.05)
    log_like_phi = np.zeros(len(grid_phi))
    gradient_phi = np.zeros(len(grid_phi))
    nat_gradient_phi = np.zeros(len(grid_phi))

    for i in range(len(grid_phi)):
        sys_model.store_params(grid_phi[i])
        pf.smoother(sys_model)
        log_like_phi[i] = pf.log_like
        gradient_phi[i] = pf.gradient_internal / (1.0 - grid_phi[i]**2)
        nat_gradient_phi[i] = pf.gradient_internal / pf.hessian_internal

    # Beta
    sys_model.create_inference_model(params_to_estimate = ('beta'))

    grid_beta = np.arange(-1, 1, 0.05)
    log_like_beta = np.zeros(len(grid_beta))
    gradient_beta = np.zeros(len(grid_beta))
    nat_gradient_beta = np.zeros(len(grid_beta))

    for i in range(len(grid_beta)):
        sys_model.store_params(grid_beta[i])
        pf.smoother(sys_model)
        log_like_beta[i] = pf.log_like
        gradient_beta[i] = pf.gradient_internal
        nat_gradient_beta[i] = pf.gradient_internal / pf.hessian_internal

    # Sigma_e
    sys_model.create_inference_model(params_to_estimate = ('sigma_e'))

    grid_sigmae = np.arange(0.1, 1, 0.05)
    log_like_sigmae = np.zeros(len(grid_sigmae))
    gradient_sigmae = np.zeros(len(grid_sigmae))
    nat_gradient_sigmae = np.zeros(len(grid_sigmae))

    for i in range(len(grid_sigmae)):
        sys_model.store_params(grid_sigmae[i])
        pf.smoother(sys_model)
        log_like_sigmae[i] = pf.log_like
        gradient_sigmae[i] = pf.gradient_internal / grid_sigmae[i]
        nat_gradient_sigmae[i] = pf.gradient_internal / pf.hessian_internal

    #Plotting
    plt.figure()
    plt.subplot(431)
    plt.plot(grid_alpha, gradient_alpha)
    plt.xlabel("alpha")
    plt.ylabel("Gradient of alpha")
    plt.axvline(x=sys_model.true_params['alpha'], color='r')
    plt.axhline(y=0.0, color='r')
    plt.subplot(432)
    plt.plot(grid_alpha, nat_gradient_alpha)
    plt.xlabel("alpha")
    plt.ylabel("Natural gradient of alpha")
    plt.axvline(x=sys_model.true_params['alpha'], color='r')
    plt.axhline(y=0.0, color='r')
    plt.subplot(433)
    plt.plot(grid_alpha, log_like_alpha)
    plt.xlabel("alpha")
    plt.ylabel("Likelihood")
    plt.axvline(x=sys_model.true_params['alpha'], color='r')

    plt.subplot(434)
    plt.plot(grid_phi, gradient_phi)
    plt.xlabel("phi")
    plt.ylabel("Gradient of phi")
    plt.axvline(x=sys_model.true_params['phi'], color='r')
    plt.axhline(y=0.0, color='r')
    plt.subplot(435)
    plt.plot(grid_phi, nat_gradient_phi)
    plt.xlabel("phi")
    plt.ylabel("Natural gradient of phi")
    plt.axvline(x=sys_model.true_params['phi'], color='r')
    plt.axhline(y=0.0, color='r')
    plt.subplot(436)
    plt.plot(grid_phi, log_like_phi)
    plt.xlabel("phi")
    plt.ylabel("Likelihood")
    plt.axvline(x=sys_model.true_params['phi'], color='r')

    plt.subplot(437)
    plt.plot(grid_beta, gradient_beta)
    plt.xlabel("beta")
    plt.ylabel("Gradient of beta")
    plt.axvline(x=sys_model.true_params['beta'], color='r')
    plt.axhline(y=0.0, color='r')
    plt.subplot(438)
    plt.plot(grid_beta, nat_gradient_beta)
    plt.xlabel("beta")
    plt.ylabel("Natural gradient of beta")
    plt.axvline(x=sys_model.true_params['beta'], color='r')
    plt.axhline(y=0.0, color='r')
    plt.subplot(439)
    plt.plot(grid_beta, log_like_beta)
    plt.xlabel("beta")
    plt.ylabel("Likelihood")
    plt.axvline(x=sys_model.true_params['beta'], color='r')

    plt.subplot(4,3,10)
    plt.plot(grid_sigmae, gradient_sigmae)
    plt.xlabel("sigma_e")
    plt.ylabel("Gradient of sigma_e")
    plt.axvline(x=sys_model.true_params['sigma_e'], color='r')
    plt.axhline(y=0.0, color='r')
    plt.subplot(4,3,11)
    plt.plot(grid_sigmae, nat_gradient_sigmae)
    plt.xlabel("sigma_e")
    plt.ylabel("Natural gradient of sigma_e")
    plt.axvline(x=sys_model.true_params['sigma_e'], color='r')
    plt.axhline(y=0.0, color='r')
    plt.subplot(4,3,12)
    plt.plot(grid_sigmae, log_like_sigmae)
    plt.xlabel("sigma_e")
    plt.ylabel("Likelihood")
    plt.axvline(x=sys_model.true_params['sigma_e'], color='r')
    plt.show()