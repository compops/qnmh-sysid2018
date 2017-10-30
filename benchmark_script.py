import numpy as np
import scripts.mh2 as mh
import scripts.qmh as qmh

mh_settings = {'no_iters': 5000,
               'no_burnin_iters': 1000,
               'step_size': 0.8,
               'base_hessian': np.eye(3) * 0.05**2,
               'initial_params': (0.0, 0.1, 0.2),
               'hessian_correction_verbose': True,
               'qn_initial_hessian': 'scaled_gradient',
               'qn_initial_hessian_scaling': 0.01,
               'verbose': False,
               'trust_region_size': 0.4,
               'qn_only_accepted_info': True,
               'qn_memory_length': 20
               }

sim_name = 'mh2'
mh.run(new_settings=mh_settings, sim_name=sim_name)

mh_settings.update({'qn_strategy': 'bfgs'})
sim_name = 'qmh_bfgs'
sim_desc = ('Damped BFGS for estimating Hessian. Scaling the initial Hessian ',
           'such that the gradient gives a step of 0.01. Non-PD estimates ',
           'are replaced with an empirical approximation of the Hessian.')
qmh.run(new_settings=mh_settings, sim_name=sim_name, sim_desc=sim_desc)

mh_settings.update({'qn_strategy': 'sr1', 'hessian_correction': 'flip'})
sim_name = 'qmh_sr1_flip'
sim_desc = ('SR1 for estimating Hessian. Scaling the initial Hessian ',
           'such that the gradient gives a step of 0.01. Non-PD estimates ',
           'are corrected by flipping negative eigenvalues.')
qmh.run(new_settings=mh_settings, sim_name=sim_name, sim_desc=sim_desc)

mh_settings.update({'qn_strategy': 'sr1', 'hessian_correction': 'replace'})
sim_name = 'qmh_sr1_hyb'
sim_desc = ('SR1 for estimating Hessian. Scaling the initial Hessian ',
           'such that the gradient gives a step of 0.01. Non-PD estimates ',
           'are replaced with an empirical approximation of the Hessian.')
qmh.run(new_settings=mh_settings, sim_name=sim_name, sim_desc=sim_desc)

mh_settings.update({'qn_strategy': 'sr1', 'hessian_correction': 'replace',
                    'qn_sr1_safe_parameterisation': True,
                    'step_size': 1.0})
sim_name = 'qmh_sr1_safe'
sim_desc = ('SR1 for estimating Hessian. Scaling the initial Hessian ',
           'such that the gradient gives a step of 0.01. Non-PD estimates ',
           'are replaced with an empirical approximation of the Hessian. ',
           'Safe parameterisation is used.')
#qmh.run(new_settings=mh_settings, sim_name=sim_name, sim_desc=sim_desc)
