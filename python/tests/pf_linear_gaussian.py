import json
import numpy as np
import matplotlib.pylab as plt
import time

from models.linear_gaussian_model import LinearGaussianModel
from state.particle_methods.standard import ParticleMethods
from state.particle_methods.cython_lgss import ParticleMethodsCythonLGSS

def run(cython_code=False, save_to_file=False):

    print("Running test script for particle smoother on the linear Gaussian " +
          "state space model.")
    # System model
    sys_model = LinearGaussianModel()
    sys_model.params['mu'] = 0.2
    sys_model.params['phi'] = 0.5
    sys_model.params['sigma_v'] = 1.0
    sys_model.params['sigma_e'] = 0.4
    sys_model.no_obs = 1000
    sys_model.initial_state = 0.0
    #sys_model.generate_data()
    #sys_model.import_data(file_name="../data/linear_gaussian_model/linear_gaussian_model_T1000_goodSNR.csv")
    sys_model.import_data(file_name="../data/linear_gaussian_model/linear_gaussian_model_T1000_midSNR.csv")

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
    #pf = ParticleMethods(particle_settings)
    pf = ParticleMethodsCythonLGSS(particle_settings)

    start_time = time.time()
    if cython_code:
        pf.flps_lgss_cython(sys_model)
    else:
        pf.smoother(sys_model)
    print("Run time of smoother:.")
    print("--- %s seconds ---" % (time.time() - start_time))

    print("MSE of filter: " + str(np.mean((pf.results['filt_state_est'][:, 0] - sys_model.states[:, 0])**2)))
    print("MSE of smoother: " + str(np.mean((pf.results['smo_state_est'][:, 0] - sys_model.states[:, 0])**2)))

    if not save_to_file:
        plt.subplot(311)
        plt.plot(np.arange(sys_model.no_obs+1), sys_model.states)
        plt.ylabel("states")
        plt.xlabel("time")
        plt.subplot(312)
        plt.plot(np.arange(sys_model.no_obs+1), np.abs(pf.results['filt_state_est'][:, 0] - sys_model.states[:, 0]))
        plt.ylabel("abs error in filtered state estimate")
        plt.xlabel("time")
        plt.title('State estimation')
        plt.subplot(313)
        plt.plot(np.arange(sys_model.no_obs+1), np.abs(pf.results['smo_state_est'][:, 0] - sys_model.states[:, 0]))
        plt.ylabel("abs error in smoothed state estimate")
        plt.xlabel("time")
        plt.title('State estimation')
        plt.show()
    return None
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

    grid_mu = np.arange(-0.95, 1, 0.10)
    log_like_mu = np.zeros((len(grid_mu), no_reps))
    gradient_mu = np.zeros((len(grid_mu), no_reps))
    nat_gradient_mu = np.zeros((len(grid_mu), no_reps))

    print("Running particle smoother over grid for mu...")
    for i in range(len(grid_mu)):
        for j in range(no_reps):
            sys_model.store_params(grid_mu[i])
            if cython_code:
                pf.flps_lgss_cython(sys_model)
            else:
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

    grid_phi = np.arange(-0.9, 1, 0.1)
    log_like_phi = np.zeros((len(grid_phi), no_reps))
    gradient_phi = np.zeros((len(grid_phi), no_reps))
    nat_gradient_phi = np.zeros((len(grid_phi), no_reps))

    print("Running particle smoother over grid for phi...")
    for i in range(len(grid_phi)):
        for j in range(no_reps):
            sys_model.store_params(grid_phi[i])
            if cython_code:
                pf.flps_lgss_cython(sys_model)
            else:
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

    grid_sigmav = np.arange(0.5, 2, 0.1)
    log_like_sigmav = np.zeros((len(grid_sigmav), no_reps))
    gradient_sigmav = np.zeros((len(grid_sigmav), no_reps))
    nat_gradient_sigmav = np.zeros((len(grid_sigmav), no_reps))

    print("Running particle smoother over grid for sigmav...")
    for i in range(len(grid_sigmav)):
        for j in range(no_reps):
            sys_model.store_params(grid_sigmav[i])
            if cython_code:
                pf.flps_lgss_cython(sys_model)
            else:
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

        file_name = "../results/particle_smoother_linear_gaussian.json"
        with open(file_name, 'w') as f:
            json.dump(output, f, ensure_ascii=False)

        print("Saved results to file: " + file_name + ".")
