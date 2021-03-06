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

import numpy as np

from models.linear_gaussian_model import LinearGaussianModel
from parameter.mcmc.metropolis_hastings import MetropolisHastings
from state.kalman_methods.standard import KalmanMethods
from state.kalman_methods.cython import KalmanMethodsCython
from state.particle_methods.standard import ParticleMethods
from state.particle_methods.cython_lgss import ParticleMethodsCythonLGSS

def run(mh_settings, cython_code=True, kf_settings=None, pf_settings=None,
        filter_method='kalman', alg_type='mh0', sim_name='test', sim_desc=".",
        seed_offset=0):

    # Set random seed for repreducibility
    np.random.seed(87655678 + int(seed_offset))

    # System model
    sys_model = LinearGaussianModel()
    sys_model.params['mu'] = 0.20
    sys_model.params['phi'] = 0.50
    sys_model.params['sigma_v'] = 1.00
    sys_model.params['sigma_e'] = 0.50
    sys_model.no_obs = 500
    sys_model.initial_state = 0.0
    sys_model.import_data(file_name="../data/linear_gaussian_model/linear_gaussian_model_T500_midSNR.csv")

    # Inference model
    sys_model.fix_true_params()
    sys_model.create_inference_model(params_to_estimate = ('mu', 'phi', 'sigma_v'))
    print(sys_model)

    # Kalman filter and smoother
    if cython_code:
        kf = KalmanMethodsCython(kf_settings)
    else:
        kf = KalmanMethods(kf_settings)

    # Particle filter and smoother
    if cython_code:
        pf = ParticleMethodsCythonLGSS(pf_settings)
    else:
        pf = ParticleMethods(pf_settings)

    # Metropolis-Hastings
    mh = MetropolisHastings(sys_model, alg_type, mh_settings)

    if filter_method is 'kalman':
        mh.run(kf)
        output_path='../results/example1'
    elif filter_method is 'particle':
        output_path='../results/example2'
        mh.run(pf)
    else:
        raise NameError("Unknown filter_method (kalman/particle).")

    # Save to file
    mh.save_to_file(output_path=output_path,
                    sim_name=sim_name,
                    sim_desc=sim_desc)