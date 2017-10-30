import numpy as np
import scripts.mh2_phillips as mh
import scripts.qmh_phillips as qmh

mh_settings = {'no_iters': 5000,
               'no_burnin_iters': 1000,
               'step_size': 0.5,
               'base_hessian': np.eye(4) * 0.02**2,
               'initial_params': (0.45, 0.76, 0.0, 0.275),
               'hessian_correction_verbose': True,
               'qn_initial_hessian': 'scaled_gradient',
               'qn_initial_hessian_scaling': 0.001,
               'verbose': False,
               'trust_region_size': 0.4,
               'qn_only_accepted_info': True,
               'qn_memory_length': 20
               }

sim_name = 'mh2_phillips'
#mh.run(new_settings=mh_settings, sim_name=sim_name)

mh_settings.update({'qn_strategy': 'bfgs'})
sim_name = 'qmh_bfgs_phillips'
sim_desc = ('Damped BFGS for estimating Hessian. Scaling the initial Hessian ',
           'such that the gradient gives a step of 0.001. Non-PD estimates ',
           'are replaced with an empirical approximation of the Hessian.')
qmh.run(new_settings=mh_settings, sim_name=sim_name, sim_desc=sim_desc)