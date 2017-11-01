
import numpy as np
import matplotlib.pylab as plt

from models.stochastic_volatility_model import SystemModel
from state.particle_methods.main import ParticleMethods
from parameter.mcmc.metropolis_hastings import MetropolisHastings

def run(new_mh_settings=None, new_kf_settings=None, new_pf_settings=None,
        sim_name='test', sim_desc='...'):
    # System model
    sys_model = SystemModel()
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
                         'estimate_hessian_segalweinstein': False,
                         'fixed_lag': 10,
                         'generate_initial_state': True
                        }
    if new_pf_settings:
        particle_settings.update(new_pf_settings)
    pf = ParticleMethods(particle_settings)

    # Metropolis-Hastings
    mh_settings = {'no_iters': 1000,
                   'no_burnin_iters': 250,
                   'step_size': 1.0,
                   'base_hessian': np.eye(3) * 0.05**2,
                   'initial_params': (0.2, 0.5, 1.0),
                   'verbose': False,
                   'hessian_correction': 'replace',
                   'qn_memory_length': 20,
                   'qn_initial_hessian': 'fixed',
                   'qn_strategy': 'bfgs',
                   'qn_bfgs_curvature_cond': 'damped', # ignore, enforce
                   'qn_initial_hessian_fixed': np.eye(3) * 0.01**2,
                   'qn_only_accepted_info': True,
                   'hessian_correction_verbose': True
                   }

    if new_mh_settings:
        mh_settings.update(new_mh_settings)
    mh = MetropolisHastings(sys_model, 'qmh', mh_settings)
    mh.run(pf)

    mh.save_to_file(output_path='../results', sim_name=sim_name, sim_desc=sim_desc)
