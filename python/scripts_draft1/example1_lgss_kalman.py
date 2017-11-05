"""Script for reproducing example 1 in paper."""
import numpy as np
import scripts_draft1.helper_linear_gaussian as mh

def main(seed_offset=0):
    """Runs the experiment."""

    # linear_gaussian_model_T1000_goodSNR
    # hessian_estimate = np.array([[0.0049,  0.0006,  0.0002 ],
    #                              [0.0006,  0.0013,  0.0002],
    #                              [0.0002,  0.0002,  0.0005 ]])

    # linear_gaussian_model_T1000_midSNR
    hessian_estimate = np.array([[  3.17e-03,  -2.65e-05,   5.843e-05],
                                 [ -2.65e-05,   1.01e-03,  -1.60e-04],
                                 [  5.84e-05,  -1.60e-04,   7.80e-04]])


    kf_settings = {'initial_state': 0.0,
                   'initial_cov': 1e-5,
                   'estimate_gradient': True
                   }

    mh_settings = {'no_iters': 20,
                   'no_burnin_iters': 10,
                   'step_size': 0.8,
                   'base_hessian': hessian_estimate,
                   'initial_params': (0.0, 0.1, 0.2),
                   'verbose': False,
                   'verbose_wait_enter': False,
                   'trust_region_size': None,
                   'hessian_estimate': None,
                   'hessian_correction': 'replace',
                   'hessian_correction_verbose': False,
                   'no_iters_between_progress_reports': 100,
                   'qn_memory_length': 20,
                   'qn_initial_hessian': 'scaled_gradient',
                   'qn_strategy': None,
                   'qn_bfgs_curvature_cond': 'damped',
                   'qn_sr1_safe_parameterisation': False,
                   'qn_sr1_skip_limit': 1e-8,
                   'qn_initial_hessian_scaling': 0.01,
                   'qn_initial_hessian_fixed': np.eye(3) * 0.01**2,
                   'qn_only_accepted_info': True
                   }

    sim_name = 'example1_mh1pre_' + str(seed_offset)
    mh.run(mh_settings=mh_settings,
           kf_settings=kf_settings,
           pf_settings=None,
           filter_method='kalman',
           alg_type='mh1',
           sim_name=sim_name,
           seed_offset=seed_offset)

    sim_name = 'example1_mh2sw_' + str(seed_offset)
    mh.run(mh_settings=mh_settings,
           kf_settings=kf_settings,
           pf_settings=None,
           filter_method='kalman',
           alg_type='mh2',
           sim_name=sim_name,
           seed_offset=seed_offset)

    mh_settings.update({'qn_strategy': 'bfgs'})
    sim_name = 'example1_mh_bfgs_' + str(seed_offset)
    sim_desc = ('Damped BFGS for estimating Hessian. Scaling the initial ',
                'Hessian such that the gradient gives a step of 0.01. Non-PD ',
                'estimates are replaced with an empirical approximation of the ',
                'Hessian.')
    mh.run(mh_settings=mh_settings,
           kf_settings=kf_settings,
           pf_settings=None,
           filter_method='kalman',
           alg_type='qmh',
           sim_name=sim_name,
           seed_offset=seed_offset)

    mh_settings.update({'qn_strategy': 'sr1',
                        'hessian_correction': 'flip'})
    sim_name = 'example1_mh_sr1_flip_' + str(seed_offset)
    sim_desc = ('SR1 for estimating Hessian. Scaling the initial Hessian ',
                'such that the gradient gives a step of 0.01. Non-PD estimates ',
                'are corrected by flipping negative eigenvalues.')

    mh.run(mh_settings=mh_settings,
           kf_settings=kf_settings,
           pf_settings=None,
           filter_method='kalman',
           alg_type='qmh',
           sim_name=sim_name,
           seed_offset=seed_offset)

    mh_settings.update({'qn_strategy': 'sr1',
                        'hessian_correction': 'replace'})
    sim_name = 'example1_mh_sr1_hyb_' + str(seed_offset)
    sim_desc = ('SR1 for estimating Hessian. Scaling the initial Hessian ',
                'such that the gradient gives a step of 0.01. Non-PD estimates ',
                'are replaced with an empirical approximation of the Hessian.')
    mh.run(mh_settings=mh_settings,
           kf_settings=kf_settings,
           pf_settings=None,
           filter_method='kalman',
           alg_type='qmh',
           sim_name=sim_name,
           seed_offset=seed_offset)

    mh_settings.update({'qn_strategy': 'bfgs',
                        'qn_only_accepted_info': False})
    sim_name = 'example1_mh_bfgs_' + str(seed_offset) + '_allinfo'
    sim_desc = ('Damped BFGS for estimating Hessian. Scaling the initial ',
                'Hessian such that the gradient gives a step of 0.01. Non-PD ',
                'estimates are replaced with an empirical approximation of the ',
                'Hessian.')
    mh.run(mh_settings=mh_settings,
           kf_settings=kf_settings,
           pf_settings=None,
           filter_method='kalman',
           alg_type='qmh',
           sim_name=sim_name,
           seed_offset=seed_offset)

    mh_settings.update({'qn_strategy': 'sr1',
                        'hessian_correction': 'flip',
                        'qn_only_accepted_info': False})
    sim_name = 'example1_mh_sr1_flip_' + str(seed_offset) + '_allinfo'
    sim_desc = ('SR1 for estimating Hessian. Scaling the initial Hessian ',
                'such that the gradient gives a step of 0.01. Non-PD estimates ',
                'are corrected by flipping negative eigenvalues.')
    mh.run(mh_settings=mh_settings,
           kf_settings=kf_settings,
           pf_settings=None,
           filter_method='kalman',
           alg_type='qmh',
           sim_name=sim_name,
           seed_offset=seed_offset)

    mh_settings.update({'qn_strategy': 'sr1',
                        'hessian_correction': 'replace',
                        'qn_only_accepted_info': False})
    sim_name = 'example1_mh_sr1_hyb_' + str(seed_offset) + '_allinfo'
    sim_desc = ('SR1 for estimating Hessian. Scaling the initial Hessian ',
                'such that the gradient gives a step of 0.01. Non-PD estimates ',
                'are replaced with an empirical approximation of the Hessian.')
    mh.run(mh_settings=mh_settings,
           kf_settings=kf_settings,
           pf_settings=None,
           filter_method='kalman',
           alg_type='qmh',
           sim_name=sim_name,
           seed_offset=seed_offset)

    return None


if __name__ == '__main__':
    main()
