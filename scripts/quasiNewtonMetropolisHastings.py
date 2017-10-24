import pickle
import uuid
import numpy as np
import matplotlib.pylab as plt

from models import linear_gaussian_model
from helpers.inference_model import get_inference_model
from state import kalman_methods
from parameter.mcmc import metropolis_hastings

#def run():
# Set random seed
np.random.seed(234878)

# System model
sys_model = linear_gaussian_model.SystemModel()
sys_model.params['mu'] = 0.20
sys_model.params['phi'] = 0.80
sys_model.params['sigma_v'] = 1.00
sys_model.params['sigma_e'] = 0.10
sys_model.no_obs = 1000
sys_model.initial_state = 0.0
#sys_model.generate_data(file_name="data/linear_gaussian_model/linear_gaussian_model_T1000_goodSNR.csv")
sys_model.import_data(file_name="data/linear_gaussian_model/linear_gaussian_model_T1000_goodSNR.csv")

# Inference model
inf_model = get_inference_model(sys_model,
                                params_to_estimate = ('mu', 'phi', 'sigma_v'))

    # Kalman filter
    kalman_settings = {'initial_state': sys_model.initial_state,
                      'initialCovariance': 1e-5,
                      'estimateGradients': True,
                      'estimateHessians': True
                      }
    kalman = kalmanMethods.FilteringSmoothing(kalman_settings)

    # Metropolis-Hastings
    quasi_newton = {
        'memory_length': 50,
        # fixed, scaled_gradient, scaled_curvature
        'initial_hessian': 'fixed',
        # bfgs, sr1
        'strategy': 'sr1',
        # ignore, damped, enforce
        'bfgs_curvature_cond': 'enforce',
        'initial_hessian_scaling': 0.10,
        'initial_hessian_fixed': 0.01**2,
        'only_accepted_info': False
    }
    mhSettings = {'no_iters': 5000,
                  'no_burnin_iters': 500,
                  'stepSize': 0.5,
                  'init_params': (0.2, 0.5, 1.0),
                  'verbose': False,
                  'waitForENTER': False,
                  'printWarningsForUnstableSystems': True,
                  'trustRegionSize': None,
                  'quasi_newton' : quasi_newton,
                  'base_hessian': np.eye(3) * 0.10**2,
                  # kalman, quasiNewton
                  'hessian_estimate': 'kalman',
                  # replace, regularise, flip
                  'hessian_correction': 'replace',
                  'hessian_correction_verbose': False
                  }

    mhSampler = metropolisHastings.ParameterEstimator(mhSettings)
    mhSampler.run(kalman, inf_model, 'mh2')
    mhSampler.plot()

    # Save run to file
    filename = str(uuid.uuid4())
    with open("runs/" + filename + ".pickle", 'wb') as f:
        pickle.dump(mhSampler, f)
    print("Saved run to " + filename + ".")
