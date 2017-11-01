import numpy as np
import scripts_draft1.mh2_stochastic_volatility as mh
import scripts_draft1.qmh_stochastic_volatility as qmh

def run(seed_offset=0):

    mh_settings = {'no_iters': 5000,
                'no_burnin_iters': 1000,
                'step_size': 0.5,
                'base_hessian': np.eye(3) * 0.01**2,
                'initial_params': (0.0, 0.9, 0.2),
                'hessian_correction_verbose': True,
                'qn_initial_hessian': 'scaled_gradient',
                'qn_initial_hessian_scaling': 0.001,
                'verbose': False,
                'trust_region_size': 0.4,
                'qn_only_accepted_info': True,
                'qn_memory_length': 20
                }

    pf_settings = {'resampling_method': 'systematic',
                'no_particles': 1000,
                'estimate_gradient': True,
                'estimate_hessian_segalweinstein': True,
                'fixed_lag': 10,
                'generate_initial_state': True
                }

    sim_name = 'example3_mh_' + str(seed_offset)
    mh.run(new_mh_settings=mh_settings,
        new_pf_settings=pf_settings,
        sim_name=sim_name,
        seed_offset=seed_offset)

    # mh_settings.update({'qn_strategy': 'bfgs'})
    # sim_name = 'example3_qmh_bfgs'
    # sim_desc = ('Damped BFGS for estimating Hessian. Scaling the initial Hessian ',
    #            'such that the gradient gives a step of 0.01. Non-PD estimates ',
    #            'are replaced with an empirical approximation of the Hessian.')
    # qmh.run(new_mh_settings=mh_settings,
    #         new_pf_settings=pf_settings,
    #         sim_name=sim_name,
    #         sim_desc=sim_desc,
    #         seed_offset=seed_offset)

    # mh_settings.update({'qn_strategy': 'sr1', 'hessian_correction': 'replace'})
    # sim_name = 'example3_qmh_sr1_hybrid'
    # sim_desc = ('SR1 for estimating Hessian. Scaling the initial Hessian ',
    #            'such that the gradient gives a step of 0.01. Non-PD estimates ',
    #            'are corrected by replacing with a global estimate.')
    # qmh.run(new_mh_settings=mh_settings,
    #         new_pf_settings=pf_settings,
    #         sim_name=sim_name,
    #         sim_desc=sim_desc,
    #         seed_offset=seed_offset)