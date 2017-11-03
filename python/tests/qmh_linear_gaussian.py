import numpy as np
import matplotlib.pylab as plt

from models.linear_gaussian_model import LinearGaussianModel
from parameter.mcmc.metropolis_hastings import MetropolisHastings
from state.particle_methods.standard import ParticleMethods

def run(alg_type='bfgs', plotting=True):
    # System model
    sys_model = LinearGaussianModel()
    sys_model.import_data(file_name="../data/linear_gaussian_model/linear_gaussian_model_T1000_goodSNR.csv")

    # Inference model
    sys_model.fix_true_params()
    sys_model.create_inference_model(params_to_estimate = ('mu', 'phi', 'sigma_v'))

    # Particle filter and smoother
    pf = ParticleMethods()
    pf.settings.update({'no_particles': 1000,
                        'fixed_lag': 10})

    # Metropolis-Hastings
    hessian_estimate = np.array([[0.00485467,  0.00062787,  0.0001611 ],
                                 [0.00062787,  0.00133698,  0.00015099],
                                 [0.0001611,   0.00015099,  0.0005252 ]])

    mh_settings = {'no_iters': 1000,
                   'no_burnin_iters': 250,
                   'step_size': 0.5,
                   'base_hessian': np.eye(3) * 0.05**2,
                   'initial_params': (0.2, 0.5, 1.0),
                   'verbose': False,
                   'hessian_correction': 'replace',
                   'qn_memory_length': 20,
                   'qn_initial_hessian': 'scaled_gradient',
                   'qn_strategy': None,
                   'qn_bfgs_curvature_cond': 'damped',
                   'qn_sr1_safe_parameterisation': False,
                   'qn_sr1_skip_limit': 1e-8,
                   'qn_initial_hessian_scaling': 0.01,
                   'qn_bfgs_curvature_cond': 'damped', # ignore, enforce
                   'qn_initial_hessian_fixed': np.eye(3) * 0.01**2,
                   'qn_only_accepted_info': True
                   }


    if alg_type is 'bfgs':
        mh_settings.update({'qn_strategy': 'bfgs'})
    elif alg_type is 'sr1':
        mh_settings.update({'qn_strategy': 'bfgs'})
    else:
        raise NameError("Unknown Quasi-Newton method...")

    mh = MetropolisHastings(sys_model, 'qmh', mh_settings)
    mh.run(pf)

    if plotting:
        mh.plot()
    else:
        mh.save_to_file(output_path='results',
                        sim_name='test_linear_gaussian_' + qmh + '_' + alg_type,
                        sim_desc='...')
