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

"""Script for reproducing example 3 in paper."""
import numpy as np
import scripts.helper_stochastic_volatility as mh

def main(seed_offset=0):
    """Runs the experiment."""

    hessian_estimate = np.array([[ 0.38292444, -0.06509644, -0.01497287, 0.0],
                                 [-0.06509644,  0.08919909, -0.04952799, 0.0],
                                 [-0.01497287, -0.04952799,  0.04612778, 0.0],
                                 [0.0, 0.0, 0.0, 0.01]])

    pf_settings = {'no_particles': 2000,
                   'resampling_method': 'systematic',
                   'fixed_lag': 10,
                   'initial_state': 0.0,
                   'generate_initial_state': True,
                   'estimate_gradient': True,
                   'estimate_hessian': True,
                   }

    mh_settings = {'no_iters': 10000,
                   'no_burnin_iters': 3000,
                   'step_size': 0.5,
                   'base_hessian': hessian_estimate,
                   'initial_params': (1.4, 0.9, 0.45, 0.0),
                   'verbose': False,
                   'verbose_wait_enter': False,
                   'trust_region_size': None,
                   'hessian_estimate': None,
                   'hessian_correction': 'replace',
                   'hessian_correction_verbose': False,
                   'no_iters_between_progress_reports': 100,
                   'qn_memory_length': 20,
                   'qn_strategy': None,
                   'qn_bfgs_curvature_cond': 'damped',
                   'qn_initial_hessian': 'scaled_gradient',
                   'qn_initial_hessian_scaling': 0.01,
                   'qn_initial_hessian_fixed': np.eye(4) * 0.01**2,
                   'qn_only_accepted_info': True,
                   'qn_accept_all_initial': True
                   }

    mh_settings.update({'qn_strategy': 'bfgs'})
    sim_name = 'example3_qmh_bfgs'
    sim_desc = ('Damped BFGS for estimating Hessian. Scaling the initial ',
                'Hessian such that the gradient gives a step of 0.01. Non-PD ',
                'estimates are replaced with an empirical approximation of ',
                'the Hessian.')
    mh.run('qmh',
            mh_settings=mh_settings,
            pf_settings=pf_settings,
            sim_name=sim_name,
            sim_desc=sim_desc,
            seed_offset=seed_offset)

    return None

if __name__ == '__main__':
    main()
