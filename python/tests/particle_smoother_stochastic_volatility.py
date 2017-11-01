import numpy as np
import matplotlib.pylab as plt
import time

from models.stochastic_volatility_model import SystemModel
from state.particle_methods.main import ParticleMethods

def run(cython_code=False):
    # System model
    sys_model = SystemModel()
    sys_model.params['mu'] = 1.45
    sys_model.params['phi'] = 0.9
    sys_model.params['sigma_v'] = 0.52
    sys_model.initial_state = 0.0

    # sys_model.import_data_quandl(handle="NASDAQOMX/OMXS30",
    #                              start_date="2012-01-02",
    #                              end_date="2014-01-02",
    #                              variable='Index Value')

    sys_model.import_data_quandl(handle="BITSTAMP/USD",
                                 start_date="2014-04-15",
                                 end_date="2017-10-30",
                                 variable='VWAP')

    # Inference model
    sys_model.fix_true_params()
    sys_model.create_inference_model(params_to_estimate = ('mu', 'phi', 'sigma_v'))

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

    # mu
    sys_model.create_inference_model(params_to_estimate = ('mu'))

    grid_mu = np.arange(0.5, 4.0, 0.1)
    log_like_mu = np.zeros(len(grid_mu))
    gradient_mu = np.zeros(len(grid_mu))
    nat_gradient_mu = np.zeros(len(grid_mu))

    for i in range(len(grid_mu)):
        sys_model.store_params(grid_mu[i])
        log_like_mu[i] = pf.log_like
        pf.smoother(sys_model)
        gradient_mu[i] = pf.gradient_internal
        nat_gradient_mu[i] = pf.gradient_internal / pf.hessian_internal

    # Phi
    sys_model.create_inference_model(params_to_estimate = ('phi'))

    grid_phi = np.arange(0.5, 1.0, 0.01)
    log_like_phi = np.zeros(len(grid_phi))
    gradient_phi = np.zeros(len(grid_phi))
    nat_gradient_phi = np.zeros(len(grid_phi))

    for i in range(len(grid_phi)):
        sys_model.store_params(grid_phi[i])
        pf.smoother(sys_model)
        log_like_phi[i] = pf.log_like
        gradient_phi[i] = pf.gradient_internal / (1.0 - grid_phi[i]**2)
        nat_gradient_phi[i] = pf.gradient_internal / pf.hessian_internal

    # sigma_v
    sys_model.create_inference_model(params_to_estimate = ('sigma_v'))

    grid_sigma_v = np.arange(0.05, 2, 0.05)
    log_like_sigma_v = np.zeros(len(grid_sigma_v))
    gradient_sigma_v = np.zeros(len(grid_sigma_v))
    nat_gradient_sigma_v = np.zeros(len(grid_sigma_v))

    for i in range(len(grid_sigma_v)):
        sys_model.store_params(grid_sigma_v[i])
        pf.smoother(sys_model)
        log_like_sigma_v[i] = pf.log_like
        gradient_sigma_v[i] = pf.gradient_internal
        nat_gradient_sigma_v[i] = pf.gradient_internal / pf.hessian_internal

    #Plotting
    plt.figure()
    plt.subplot(331)
    plt.plot(grid_mu, gradient_mu)
    plt.xlabel("mu")
    plt.ylabel("Gradient of mu")
    plt.axvline(x=sys_model.true_params['mu'], color='r')
    plt.axhline(y=0.0, color='r')
    plt.subplot(332)
    plt.plot(grid_mu, nat_gradient_mu)
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
    plt.plot(grid_phi, nat_gradient_phi)
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
    plt.plot(grid_sigma_v, gradient_sigma_v)
    plt.xlabel("sigma_v")
    plt.ylabel("Gradient of sigma_v")
    plt.axvline(x=sys_model.true_params['sigma_v'], color='r')
    plt.axhline(y=0.0, color='r')
    plt.subplot(338)
    plt.plot(grid_sigma_v, nat_gradient_sigma_v)
    plt.xlabel("sigma_v")
    plt.ylabel("Natural gradient of sigma_v")
    plt.axvline(x=sys_model.true_params['sigma_v'], color='r')
    plt.axhline(y=0.0, color='r')
    plt.subplot(339)
    plt.plot(grid_sigma_v, log_like_sigma_v)
    plt.xlabel("sigma_v")
    plt.ylabel("Likelihood")
    plt.axvline(x=sys_model.true_params['sigma_v'], color='r')
    plt.show()