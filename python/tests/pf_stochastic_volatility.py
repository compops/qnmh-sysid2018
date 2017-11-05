import numpy as np
import matplotlib.pylab as plt
import time
import json

from models.stochastic_volatility_model import StochasticVolatilityModel
from state.particle_methods.standard import ParticleMethods

def run(cython_code=False, save_to_file=False):
    # System model
    sys_model = StochasticVolatilityModel()
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
                         'estimate_hessian': True,
                         'fixed_lag': 10,
                         'generate_initial_state': True
                        }
    pf = ParticleMethods(particle_settings)
    pf.smoother(sys_model)

    if not save_to_file:
        plt.subplot(211)
        plt.plot(np.arange(sys_model.no_obs+1), pf.results['filt_state_est'][:, 0])
        plt.ylabel("filtered state estimate")
        plt.xlabel("time")
        plt.title('State estimation')

        plt.subplot(212)
        plt.plot(np.arange(sys_model.no_obs+1), pf.results['smo_state_est'][:, 0])
        plt.ylabel("smoothed state estimate")
        plt.xlabel("time")
        plt.title('State estimation')
        plt.show()

    if save_to_file:
        repetitions = 100
    else:
        repetitions = 1

    # mu
    print("Running particle smoother over grid for mu...")
    sys_model.create_inference_model(params_to_estimate = ('mu'))

    grid_mu = np.arange(0.5, 4.0, 0.1)
    log_like_mu = np.zeros((len(grid_mu), repetitions))
    gradient_mu = np.zeros((len(grid_mu), repetitions))
    nat_gradient_mu = np.zeros((len(grid_mu), repetitions))

    for i in range(len(grid_mu)):
        for j in range(repetitions):
            sys_model.store_params(grid_mu[i])
            pf.smoother(sys_model)
            log_like_mu[i, j] = pf.results['log_like'].flatten()
            gradient_mu[i, j] = pf.results['gradient_internal'].flatten()
            nat_gradient_mu[i, ] = pf.results['gradient_internal'].flatten()
            nat_gradient_mu[i, ] /= pf.results['hessian_internal'].flatten()
            print("Grid point: {}/{} and iteration {}/{}".format(i,
                  len(grid_mu), j, repetitions))

    # Phi
    print("Running particle smoother over grid for phi...")
    sys_model.create_inference_model(params_to_estimate = ('phi'))

    grid_phi = np.arange(0.5, 1.0, 0.01)
    log_like_phi = np.zeros((len(grid_phi), repetitions))
    gradient_phi = np.zeros((len(grid_phi), repetitions))
    nat_gradient_phi = np.zeros((len(grid_phi), repetitions))

    for i in range(len(grid_phi)):
        for j in range(repetitions):
            sys_model.store_params(grid_phi[i])
            pf.smoother(sys_model)
            log_like_phi[i, j] = pf.results['log_like'].flatten()
            gradient_phi[i, j] = pf.results['gradient_internal'].flatten()
            gradient_phi[i, j]/= (1.0 - grid_phi[i]**2)
            nat_gradient_phi[i, j] = pf.results['gradient_internal'].flatten()
            nat_gradient_phi[i, j]/= pf.results['pf.hessian_internal'].flatten()
            print("Grid point: {}/{} and iteration {}/{}".format(i,
                  len(grid_phi), j, repetitions))

    # sigma_v
    print("Running particle smoother over grid for sigma_v...")
    sys_model.create_inference_model(params_to_estimate = ('sigma_v'))

    grid_sigmav = np.arange(0.05, 2, 0.05)
    log_like_sigmav = np.zeros((len(grid_sigmav), repetitions))
    gradient_sigmav = np.zeros((len(grid_sigmav), repetitions))
    nat_gradient_sigmav = np.zeros((len(grid_sigmav), repetitions))

    for i in range(len(grid_sigmav)):
        for j in range(repetitions):
            sys_model.store_params(grid_sigmav[i])
            pf.smoother(sys_model)
            log_like_sigmav[i, j] = pf.results['log_like;'].flatten()
            gradient_sigmav[i, j] = pf.results['gradient_internal'].flatten()
            gradient_sigmav[i, j] /= grid_sigmav[i]
            nat_gradient_sigmav[i, j] = pf.results['gradient_internal'].flatten()
            nat_gradient_sigmav[i, j] /= pf.results['pf.hessian_internal'].flatten()
            print("Grid point: {}/{} and iteration {}/{}".format(i,
                  len(grid_sigmav), j, repetitions))

    #Plotting
    if not save_to_file:
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
        plt.plot(grid_sigmav, gradient_sigmav)
        plt.xlabel("sigma_v")
        plt.ylabel("Gradient of sigma_v")
        plt.axvline(x=sys_model.true_params['sigma_v'], color='r')
        plt.axhline(y=0.0, color='r')
        plt.subplot(338)
        plt.plot(grid_sigmav, nat_gradient_sigmav)
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

    # Save the results
    if save_to_file:
        output = {'grid_mu': grid_mu,
                  'gradient_mu': gradient_mu,
                  'nat_gradient_mu': nat_gradient_mu,
                  'grid_phi': grid_phi,
                  'gradient_phi': gradient_phi,
                  'nat_gradient_phi': nat_gradient_phi,
                  'grid_sigmav': grid_sigmav,
                  'gradient_sigmav': gradient_sigmav,
                  'nat_gradient_sigmav': nat_gradient_sigmav
                  }

        for key in output:
            if isinstance(output[key], np.ndarray):
                output[key] = output[key].tolist()

        with open("particle_smoother_stochastic_volatility.json", 'w') as f:
            json.dump(output, f, ensure_ascii=False)