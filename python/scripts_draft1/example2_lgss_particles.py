"""Script for reproducing example 2 in paper."""
import numpy as np
import scripts_draft1.helper_linear_gaussian as mh

def main(seed_offset=0):
    """Runs the experiment."""

    mh_settings = {'no_iters': 10000,
                   'no_burnin_iters': 3000,
                   'step_size': 1.0,
                   'base_hessian': np.eye(3) * 0.05**2,
                   'initial_params': (0.0, 0.1, 0.2),
                   'verbose': False,
                   'verbose_wait_enter': False,
                   'trust_region_size': None,
                   'hessian_estimate': None,
                   'hessian_correction': 'replace',
                   'hessian_correction_verbose': False,
                   'no_iters_between_progress_reports': 100,
                   'qn_memory_length': 20,
                   'qn_initial_hessian': 'fixed',
                   'qn_strategy': None,
                   'qn_bfgs_curvature_cond': 'ignore',
                   'qn_sr1_safe_parameterisation': False,
                   'qn_sr1_skip_limit': 1e-8,
                   'qn_initial_hessian_scaling': 0.10,
                   'qn_initial_hessian_fixed': np.eye(3) * 0.01**2,
                   'qn_only_accepted_info': True
                   }

    pf_settings = {'no_particles': 1000,
                   'resampling_method': 'systematic',
                   'fixed_lag': 10,
                   'initial_state': 0,0,
                   'generate_initial_state': True,
                   'estimate_gradient': True,
                   'estimate_hessian': True,
                   }

    sim_name = 'example2_mh2_' + str(seed_offset)
    mh.run('mh2',
           mh_settings=mh_settings,
           kf_settings=None,
           pf_settings=pf_settings,
           smoothing_method="particle",
           sim_name=sim_name,
           seed_offset=seed_offset)

    mh_settings.update({'qn_strategy': 'bfgs'})
    sim_name = 'example2_qmh_bfgs_' + str(seed_offset)
    sim_desc = ('Damped BFGS for estimating Hessian. Scaling the initial ',
                'Hessian such that the gradient gives a step of 0.01. Non-PD ',
                'estimates are replaced with an empirical approximation of ',
                ' the Hessian.')
    mh.run('qmh',
           mh_settings=mh_settings,
           kf_settings=None,
           pf_settings=pf_settings,
           smoothing_method="particle",
          sim_name=sim_name,
           sim_desc=sim_desc,
           seed_offset=seed_offset)

    mh_settings.update({'qn_strategy': 'sr1',
                        'hessian_correction': 'flip'})
    sim_name = 'example2_qmh_sr1_flip_' + str(seed_offset)
    sim_desc = ('SR1 for estimating Hessian. Scaling the initial Hessian ',
                'such that the gradient gives a step of 0.01. Non-PD ',
                'estimates are corrected by flipping negative eigenvalues.')
    mh.run('qmh',,
           mh_settings=mh_settings,
           kf_settings=None,
           pf_settings=pf_settings,
           smoothing_method="particle",
           sim_name=sim_name,
           sim_desc=sim_desc,
           seed_offset=seed_offset)

    mh_settings.update({'qn_strategy': 'sr1',
                        'hessian_correction': 'replace'})
    sim_name = 'example2_qmh_sr1_hyb_' + str(seed_offset)
    sim_desc = ('SR1 for estimating Hessian. Scaling the initial Hessian ',
                'such that the gradient gives a step of 0.01. Non-PD estimates ',
                'are replaced with an empirical approximation of the Hessian.')
    mh.run('qmh',
           mh_settings=mh_settings,
           kf_settings=None,
           pf_settings=pf_settings,
           smoothing_method="particle",
           sim_name=sim_name,
           sim_desc=sim_desc,
           seed_offset=seed_offset)

    return None

if __name__ == '__main__':
    main()
