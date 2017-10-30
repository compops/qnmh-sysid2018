import numpy as np
import matplotlib.pylab as plt

from models.linear_gaussian_model import SystemModel
from state.kalman_methods.main import KalmanMethods
from state.particle_methods.main import ParticleMethods
from parameter.mcmc.metropolis_hastings import MetropolisHastings

def run(new_mh_settings=None, new_kf_settings=None, new_pf_settings=None,
        smoothing_method="kalman", sim_name='test', sim_desc='...'):
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
    kf_settings = {'initial_state': sys_model.initial_state}
    if new_kf_settings:
        kf_settings.update(new_kf_settings)
    kf = KalmanMethods(kf_settings)

    # Particle filter and smoother
    particle_settings = {'resampling_method': 'systematic',
                         'no_particles': 1000,
                         'estimate_gradient': True,
                         'estimate_hessian_segalweinstein': True,
                         'fixed_lag': 10,
                         'generate_initial_state': True
                        }
    if new_pf_settings:
        particle_settings.update(new_pf_settings)
    pf = ParticleMethods(particle_settings)

    # Metropolis-Hastings
    mh_settings = {'no_iters': 1000,
                   'no_burnin_iters': 250,
                   'step_size': 0.5,
                   'base_hessian': np.eye(3) * 0.05**2,
                   'initial_params': (0.2, 0.5, 1.0),
                   'verbose': False,
                   'hessian_estimate': 'kalman',
                   'hessian_correction_verbose': True
                   }

    if new_mh_settings:
        mh_settings.update(new_mh_settings)
    mh = MetropolisHastings(sys_model, 'mh2', mh_settings)

    if smoothing_method is 'kalman':
        mh.run(kf)
    elif smoothing_method is 'particle':
        mh.run(pf)
    else:
        raise NameError("Unknown smoothing method selected")

    mh.save_to_file(output_path='results', sim_name=sim_name, sim_desc=sim_desc)