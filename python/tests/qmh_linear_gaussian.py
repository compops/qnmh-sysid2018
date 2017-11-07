import numpy as np
import matplotlib.pylab as plt

from models.linear_gaussian_model import LinearGaussianModel
from parameter.mcmc.metropolis_hastings import MetropolisHastings
from state.particle_methods.standard import ParticleMethods
from state.particle_methods.cython_lgss import ParticleMethodsCythonLGSS
from state.kalman_methods.standard import KalmanMethods
from state.kalman_methods.cython_code import KalmanMethodsCython

def run(cython_code=True, filter_method='kalman', alg_type='bfgs',
        plotting=True, file_tag=None, **kwargs):

    # System model
    sys_model = LinearGaussianModel()
    sys_model.params['mu'] = 0.20
    sys_model.params['phi'] = 0.50
    sys_model.params['sigma_v'] = 1.00
    sys_model.params['sigma_e'] = 0.40
    sys_model.no_obs = 1000
    sys_model.initial_state = 0.0
    #sys_model.import_data(file_name="../data/linear_gaussian_model/linear_gaussian_model_T1000_goodSNR.csv")
    sys_model.import_data(file_name="../data/linear_gaussian_model/linear_gaussian_model_T1000_midSNR.csv")

    # Inference model
    sys_model.fix_true_params()
    sys_model.create_inference_model(params_to_estimate = ('mu', 'phi', 'sigma_v'))

    # Kalman filter and smoother
    if cython_code:
        kf = KalmanMethodsCython()
    else:
        kf = KalmanMethods()
    if kwargs:
        kf.settings.update(kwargs)

    # Particle filter and smoother
    if cython_code:
        pf = ParticleMethodsCythonLGSS()
    else:
        pf = ParticleMethods()
    pf.settings.update({'no_particles': 1000, 'fixed_lag': 10})
    if kwargs:
        pf.settings.update(kwargs)

    # Metropolis-Hastings
    # linear_gaussian_model_T1000_goodSNR
    # hessian_estimate = np.array([[0.00485467,  0.00062787,  0.0001611 ],
    #                              [0.00062787,  0.00133698,  0.00015099],
    #                              [0.0001611,   0.00015099,  0.0005252 ]])

    # linear_gaussian_model_T1000_midSNR
    hessian_estimate = np.array([[  3.17466496e-03,  -2.65148861e-05,   5.84256527e-05],
                                 [ -2.65148861e-05,   1.00771014e-03,  -1.59533168e-04],
                                 [  5.84256527e-05,  -1.59533168e-04,   7.80308724e-04]])


    mh_settings = {'no_iters': 2500,
                   'no_burnin_iters': 500,
                   'step_size': 0.8,
                   'base_hessian': hessian_estimate,
                   'initial_params': (0.2, 0.5, 1.0),
                   'verbose': False,
                   'hessian_correction': 'replace',
                   'qn_memory_length': 50,
                   'qn_strategy': None,
                   'qn_bfgs_curvature_cond': 'damped',
                   'qn_sr1_safe_parameterisation': False,
                   'qn_sr1_skip_limit': 1e-8,
                   'qn_initial_hessian': 'scaled_gradient',
                   'qn_initial_hessian_scaling': 0.01,
                   'qn_initial_hessian_fixed': np.eye(3) * 0.01**2,
                   'qn_only_accepted_info': True,
                   'qn_accept_all_initial': False
                   }


    if alg_type is 'bfgs':
        mh_settings.update({'qn_strategy': 'bfgs'})
    elif alg_type is 'sr1':
        mh_settings.update({'qn_strategy': 'sr1'})
    else:
        raise NameError("Unknown Quasi-Newton method...")

    if kwargs:
        mh_settings.update(kwargs)
    mh = MetropolisHastings(sys_model, 'qmh', mh_settings)

    if filter_method is 'kalman':
        mh.run(kf)
    elif filter_method is 'particle':
        mh.run(pf)
    else:
        raise NameError("Unknown filter_method (kalman/particle).")

    if plotting:
        mh.plot()
    else:
        sim_name = 'test_linear_gaussian_' + filter_method + '_' + 'qmh_' + alg_type
        if file_tag:
            sim_name += '_' + file_tag
                mh.save_to_file(output_path='../results-tests/qmh-linear-gaussian/',
                        sim_name=sim_name,
                        sim_desc='...')
