###############################################################################
#    Constructing Metropolis-Hastings proposals using damped BFGS updates
#    Copyright (C) 2018  Johan Dahlin < uni (at) johandahlin [dot] com >
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
###############################################################################

"""Script for reproducing example 1 in paper."""
import numpy as np
import scripts.helper_linear_gaussian as mh

def main(cython_code=True, seed_offset=0):
    """Runs the experiment."""

    hessian_estimate = np.array([[ 0.00397222, -0.00228247,  0.00964908],
                                 [-0.00228247,  0.00465944, -0.00961161],
                                 [ 0.00964908, -0.00961161,  0.05049018]])

    kf_settings = {'initial_state': 0.0,
                   'initial_cov': 1e-5,
                   'estimate_gradient': True
                   }

    mh_settings = {'no_iters': 10000,
                   'no_burnin_iters': 3000,
                   'step_size': 0.5,
                   'base_hessian': hessian_estimate,
                   'initial_params': (0.2, 0.5, 1.0),
                   'verbose': False,
                   'verbose_wait_enter': False,
                   'trust_region_size': None,
                   'hessian_estimate': None,
                   'hessian_correction': 'replace',
                   'hessian_correction_verbose': False,
                   'no_iters_between_progress_reports': 1000,
                   'qn_memory_length': 20,
                   'qn_initial_hessian': 'scaled_gradient',
                   'qn_strategy': None,
                   'qn_bfgs_curvature_cond': 'damped',
                   'qn_initial_hessian_scaling': 0.01,
                   'qn_initial_hessian_fixed': np.eye(3) * 0.01**2,
                   'qn_only_accepted_info': True,
                   'qn_accept_all_initial': True
                   }

    mh_settings.update({'step_size': 2.38 / np.sqrt(3)})
    sim_name = 'example1_mh0pre_' + str(seed_offset)
    mh.run(mh_settings=mh_settings,
           cython_code=True,
           kf_settings=kf_settings,
           pf_settings=None,
           filter_method='kalman',
           alg_type='mh0',
           sim_name=sim_name,
           seed_offset=seed_offset)

    mh_settings.update({'step_size': 0.5 * 1.38 / np.sqrt(3**(1/3))})
    sim_name = 'example1_mh1pre_' + str(seed_offset)
    mh.run(mh_settings=mh_settings,
           cython_code=True,
           kf_settings=kf_settings,
           pf_settings=None,
           filter_method='kalman',
           alg_type='mh1',
           sim_name=sim_name,
           seed_offset=seed_offset)

    mh_settings.update({'step_size': 0.5})
    mh_settings.update({'base_hessian': 0.01**2 * np.eye(3)})
    sim_name = 'example1_mh2sw_' + str(seed_offset)
    mh.run(mh_settings=mh_settings,
           cython_code=True,
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
           cython_code=True,
           kf_settings=kf_settings,
           pf_settings=None,
           filter_method='kalman',
           alg_type='qmh',
           sim_name=sim_name,
           seed_offset=seed_offset)

    mh_settings.update({'qn_strategy': 'bfgs',
                        'qn_bfgs_curvature_cond': 'enforce',
                        'hessian_correction': 'replace'})
    sim_name = 'example1_mh_bfgs_' + str(seed_offset) + '_enforce_replace'
    sim_desc = ('Damped BFGS for estimating Hessian. Scaling the initial ',
                'Hessian such that the gradient gives a step of 0.01. Non-PD ',
                'estimates are replaced with an empirical approximation of the ',
                'Hessian.')
    mh.run(mh_settings=mh_settings,
           cython_code=True,
           kf_settings=kf_settings,
           pf_settings=None,
           filter_method='kalman',
           alg_type='qmh',
           sim_name=sim_name,
           seed_offset=seed_offset)

    mh_settings.update({'qn_strategy': 'bfgs',
                        'qn_bfgs_curvature_cond': 'ignore',
                        'hessian_correction': 'replace'})
    sim_name = 'example1_mh_bfgs_' + str(seed_offset) + '_ignore_replace'
    sim_desc = ('Damped BFGS for estimating Hessian. Scaling the initial ',
                'Hessian such that the gradient gives a step of 0.01. Non-PD ',
                'estimates are replaced with an empirical approximation of the ',
                'Hessian.')
    mh.run(mh_settings=mh_settings,
           cython_code=True,
           kf_settings=kf_settings,
           pf_settings=None,
           filter_method='kalman',
           alg_type='qmh',
           sim_name=sim_name,
           seed_offset=seed_offset)

    mh_settings.update({'qn_strategy': 'bfgs',
                        'qn_bfgs_curvature_cond': 'ignore',
                        'hessian_correction': 'regularise'})
    sim_name = 'example1_mh_bfgs_' + str(seed_offset) + '_ignore_reg'
    sim_desc = ('Damped BFGS for estimating Hessian. Scaling the initial ',
                'Hessian such that the gradient gives a step of 0.01. Non-PD ',
                'estimates are replaced with an empirical approximation of the ',
                'Hessian.')
    mh.run(mh_settings=mh_settings,
           cython_code=True,
           kf_settings=kf_settings,
           pf_settings=None,
           filter_method='kalman',
           alg_type='qmh',
           sim_name=sim_name,
           seed_offset=seed_offset)

    mh_settings.update({'qn_strategy': 'bfgs',
                        'qn_bfgs_curvature_cond': 'ignore',
                        'hessian_correction': 'flip'})
    sim_name = 'example1_mh_bfgs_' + str(seed_offset) + '_ignore_flip'
    sim_desc = ('Damped BFGS for estimating Hessian. Scaling the initial ',
                'Hessian such that the gradient gives a step of 0.01. Non-PD ',
                'estimates are replaced with an empirical approximation of the ',
                'Hessian.')
    mh.run(mh_settings=mh_settings,
           cython_code=True,
           kf_settings=kf_settings,
           pf_settings=None,
           filter_method='kalman',
           alg_type='qmh',
           sim_name=sim_name,
           seed_offset=seed_offset)

    return None


if __name__ == '__main__':
    main()
