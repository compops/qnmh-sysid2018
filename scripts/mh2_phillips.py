import numpy as np
import matplotlib.pylab as plt

from models.phillips_curve_model import SystemModel
from state.particle_methods.main import ParticleMethods
from parameter.mcmc.metropolis_hastings import MetropolisHastings

def run(new_settings=None, sim_name='test', sim_desc='...'):
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

    # Metropolis-Hastings
    mh_settings = {'no_iters': 1000,
                   'no_burnin_iters': 250,
                   'step_size': 0.5,
                   'base_hessian': np.eye(4) * 0.02**2,
                   'initial_params': (0.45, 0.76, 0.0, 0.275),
                   'verbose': False,
                   'hessian_estimate': 'kalman',
                   'hessian_correction_verbose': True
                   }

    if new_settings:
        mh_settings.update(new_settings)

    mh = MetropolisHastings(sys_model, 'mh2', mh_settings)
    mh.run(pf)
    mh.save_to_file(output_path='results', sim_name=sim_name, sim_desc=sim_desc)