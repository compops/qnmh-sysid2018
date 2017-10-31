import numpy as np
import scripts.qmh_stochastic_volatility as qmh

mh_settings = {'no_iters': 5000,
               'no_burnin_iters': 1000,
               'step_size': 0.8,
               'base_hessian': np.eye(3) * 0.01**2,
               'initial_params': (0.0, 0.9, 0.2),
               'hessian_correction_verbose': True,
               'qn_initial_hessian': 'scaled_gradient',
               'qn_initial_hessian_scaling': 0.01,
               'verbose': False,
               'trust_region_size': 0.4,
               'qn_only_accepted_info': True,
               'qn_memory_length': 20
               }

pf_settings = {'resampling_method': 'systematic',
               'no_particles': 1000,
               'estimate_gradient': True,
               'estimate_hessian_segalweinstein': False,
               'fixed_lag': 10,
               'generate_initial_state': True
              }

mh_settings.update({'qn_strategy': 'bfgs'})
sim_name = 'example3_qmh_bfgs'
sim_desc = ('Damped BFGS for estimating Hessian. Scaling the initial Hessian ',
           'such that the gradient gives a step of 0.01. Non-PD estimates ',
           'are replaced with an empirical approximation of the Hessian.')
qmh.run(new_mh_settings=mh_settings, new_pf_settings=pf_settings,
        sim_name=sim_name, sim_desc=sim_desc)

# mh_settings.update({'qn_strategy': 'sr1', 'hessian_correction': 'flip'})
# sim_name = 'example2_qmh_sr1_flip'
# sim_desc = ('SR1 for estimating Hessian. Scaling the initial Hessian ',
#            'such that the gradient gives a step of 0.01. Non-PD estimates ',
#            'are corrected by flipping negative eigenvalues.')
# qmh.run(new_mh_settings=mh_settings, new_pf_settings=pf_settings,
#         sim_name=sim_name, sim_desc=sim_desc)