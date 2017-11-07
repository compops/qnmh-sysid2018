import json
import numpy as np
import matplotlib.pylab as plt
import time

from models.stochastic_volatility_model import StochasticVolatilityModel
from state.particle_methods.standard import ParticleMethods
from state.particle_methods.cython_sv import ParticleMethodsCythonSV

def run(cython_code=False, save_to_file=False):

    print("Running test script for particle smoother on the stochastic " +
          "volatility state space model.")
    # System model
    sys_model = StochasticVolatilityModel()
    sys_model.params['mu'] = 1.4
    sys_model.params['phi'] = 0.9
    sys_model.params['sigma_v'] = 0.6
    sys_model.initial_state = 0.0

    # sys_model.import_data_quandl(handle="NASDAQOMX/OMXS30",
    #                              start_date="2012-01-02",
    #                              end_date="2014-01-02",
    #                              variable='Index Value')

    sys_model.import_data_quandl(handle="BITSTAMP/USD",
                                 start_date="2015-11-07",
                                 end_date="2017-11-07",
                                 variable='VWAP',
                                 api_key="LWnsxBRpquFe9fWcanPF")

    # Inference model
    sys_model.fix_true_params()
    sys_model.create_inference_model(params_to_estimate = ('mu', 'phi', 'sigma_v'))

    # Kalman filter and smoother
    particle_settings = {'resampling_method': 'systematic',
                         'no_particles': 2000,
                         'estimate_gradient': True,
                         'fixed_lag': 10,
                         'estimate_hessian': True,
                         'generate_initial_state': True,
                         'initial_state': 0.0
                        }
    if cython_code:
        pf = ParticleMethodsCythonSV(particle_settings)
    else:
        pf = ParticleMethods(particle_settings)

    start_time = time.time()
    pf.smoother(sys_model)
    print("Run time of smoother:.")
    print("--- %s seconds ---" % (time.time() - start_time))

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
        no_reps = 100
    else:
        no_reps = 1

    if save_to_file:
        print("Running particle filter to estimate StDev of log-like estimator")
        log_like_ests = np.zeros(no_reps)
        for i in range(no_reps):
            pf.filter(sys_model)
            log_like_ests[i] = pf.results['log_like']
            print("Iteration {}/{}".format(i, no_reps))
        print("Mean of log-likelihood estimate: {}".format(np.mean(log_like_ests)))
        print("StDev of log-likelihood estimate: {}".format(np.sqrt(np.var(log_like_ests))))

    # Mu
    sys_model.create_inference_model(params_to_estimate = ('mu'))

    grid_mu = np.arange(0.0, 3.0, 0.10)
    log_like_mu = np.zeros((len(grid_mu), no_reps))
    gradient_mu = np.zeros((len(grid_mu), no_reps))
    nat_gradient_mu = np.zeros((len(grid_mu), no_reps))

    print("Running particle smoother over grid for mu...")
    for i in range(len(grid_mu)):
        for j in range(no_reps):
            sys_model.store_params(grid_mu[i])
            pf.smoother(sys_model)
            log_like_mu[i, j] = pf.results['log_like']
            gradient_mu[i, j] = pf.results['gradient_internal'].flatten()
            gradient_mu[i, j] /= (1.0 - sys_model.params['phi'])
            nat_gradient_mu[i, j] = pf.results['gradient_internal'].flatten()
            nat_gradient_mu[i, j] /=  pf.results['hessian_internal'].flatten()
            print("Grid point: {}/{} and iteration {}/{}".format(i,
                  len(grid_mu), j, no_reps))


    # Phi
    sys_model.create_inference_model(params_to_estimate = ('phi'))

    grid_phi = np.arange(0.0, 1, 0.05)
    log_like_phi = np.zeros((len(grid_phi), no_reps))
    gradient_phi = np.zeros((len(grid_phi), no_reps))
    nat_gradient_phi = np.zeros((len(grid_phi), no_reps))

    print("Running particle smoother over grid for phi...")
    for i in range(len(grid_phi)):
        for j in range(no_reps):
            sys_model.store_params(grid_phi[i])
            pf.smoother(sys_model)
            log_like_phi[i, j] = pf.results['log_like']
            gradient_phi[i, j] = pf.results['gradient_internal'].flatten()
            gradient_phi[i, j] /= (1.0 - grid_phi[i]**2)
            nat_gradient_phi[i, j] = pf.results['gradient_internal'].flatten()
            nat_gradient_phi[i, j] /= pf.results['hessian_internal'].flatten()
            print("Grid point: {}/{} and iteration {}/{}".format(i,
                  len(grid_phi), j, no_reps))

    # Sigma_v
    sys_model.create_inference_model(params_to_estimate = ('sigma_v'))

    grid_sigmav = np.arange(0.05, 1.0, 0.05)
    log_like_sigmav = np.zeros((len(grid_sigmav), no_reps))
    gradient_sigmav = np.zeros((len(grid_sigmav), no_reps))
    nat_gradient_sigmav = np.zeros((len(grid_sigmav), no_reps))

    print("Running particle smoother over grid for sigmav...")
    for i in range(len(grid_sigmav)):
        for j in range(no_reps):
            sys_model.store_params(grid_sigmav[i])
            pf.smoother(sys_model)
            log_like_sigmav[i, j] = pf.results['log_like']
            gradient_sigmav[i, j] = pf.results['gradient_internal'].flatten()
            gradient_sigmav[i, j] /= grid_sigmav[i]
            nat_gradient_sigmav[i, j] = pf.results['gradient_internal'].flatten()
            nat_gradient_sigmav[i, j] /= pf.results['hessian_internal'].flatten()
            print("Grid point: {}/{} and iteration {}/{}".format(i,
                  len(grid_sigmav), j, no_reps))


    # Plotting
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
                  'log_like_mu': log_like_mu,
                  'gradient_mu': gradient_mu,
                  'nat_gradient_mu': nat_gradient_mu,
                  'grid_phi': grid_phi,
                  'log_like_phi': log_like_phi,
                  'gradient_phi': gradient_phi,
                  'nat_gradient_phi': nat_gradient_phi,
                  'grid_sigmav': grid_sigmav,
                  'log_like_sigmav': log_like_sigmav,
                  'gradient_sigmav': gradient_sigmav,
                  'nat_gradient_sigmav': nat_gradient_sigmav
                  }

        for key in output:
            if isinstance(output[key], np.ndarray):
                output[key] = output[key].tolist()

        file_name = "../results-tests/particle_smoother_stochastic_volatility.json"
        with open(file_name, 'w') as f:
            json.dump(output, f, ensure_ascii=False)

        print("Saved results to file: " + file_name + ".")
