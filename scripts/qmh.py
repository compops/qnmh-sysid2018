import numpy as np
import matplotlib.pylab as plt

from models.linear_gaussian_model import SystemModel
from parameter.mcmc.metropolis_hastings import MetropolisHastings
from state.kalman_methods import KalmanMethods

def run(new_settings=None, sim_name='test', sim_desc='...'):
    # System model
    sys_model = SystemModel()
    sys_model.params['mu'] = 0.20
    sys_model.params['phi'] = 0.50
    sys_model.params['sigma_v'] = 1.00
    sys_model.params['sigma_e'] = 0.10
    sys_model.no_obs = 1000
    sys_model.initial_state = 0.0

    #sys_model.generate_data(file_name="data/linear_gaussian_model/linear_gaussian_model_T1000_goodSNR.csv")
    sys_model.import_data(file_name="data/linear_gaussian_model/linear_gaussian_model_T1000_goodSNR.csv")

    # Inference model
    sys_model.fix_true_params()
    sys_model.create_inference_model(params_to_estimate = ('mu', 'phi', 'sigma_v'))

    # Kalman filter and smoother
    kf_settings = {'initial_state': sys_model.initial_state}
    kf = KalmanMethods(kf_settings)

    # Metropolis-Hastings
    mh_settings = {'no_iters': 1000,
                   'no_burnin_iters': 250,
                   'step_size': 0.5,
                   'base_hessian': np.eye(3) * 0.05**2,
                   'initial_params': (0.2, 0.5, 1.0),
                   'verbose': False,
                   'hessian_correction': 'replace',
                   'qn_memory_length': 20,
                   'qn_initial_hessian': 'fixed',
                   'qn_strategy': 'bfgs',
                   'qn_bfgs_curvature_cond': 'damped', # ignore, enforce
                   'qn_initial_hessian_fixed': np.eye(3) * 0.01**2,
                   'qn_only_accepted_info': True
                   }

    if new_settings:
        mh_settings.update(new_settings)

    mh = MetropolisHastings(sys_model, 'qmh', mh_settings)
    mh.run(kf)
    mh.save_to_file(output_path='results', sim_name=sim_name, sim_desc=sim_desc)
