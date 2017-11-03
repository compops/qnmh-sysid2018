import numpy as np
import matplotlib.pylab as plt

from models.linear_gaussian_model import LinearGaussianModel
from parameter.mcmc.metropolis_hastings import MetropolisHastings
from state.kalman_methods.standard import KalmanMethods
from state.particle_methods.standard import ParticleMethods

def run(filter_method='kalman', alg_type='mh0', plotting=True):
    # System model
    sys_model = LinearGaussianModel()
    sys_model.params['mu'] = 0.20
    sys_model.params['phi'] = 0.50
    sys_model.params['sigma_v'] = 1.00
    sys_model.params['sigma_e'] = 0.40
    sys_model.no_obs = 1000
    sys_model.initial_state = 0.0
    #sys_model.generate_data(file_name="../data/linear_gaussian_model/linear_gaussian_model_T1000_midSNR.csv")
    sys_model.import_data(file_name="../data/linear_gaussian_model/linear_gaussian_model_T1000_midSNR.csv")
    #sys_model.import_data(file_name="../data/linear_gaussian_model/linear_gaussian_model_T1000_goodSNR.csv")

    # Inference model
    sys_model.fix_true_params()
    sys_model.create_inference_model(params_to_estimate = ('mu', 'phi', 'sigma_v'))
    print(sys_model)

    # Kalman filter and smoother
    kf = KalmanMethods()

    # Particle filter and smoother
    pf = ParticleMethods()
    pf.settings.update({'no_particles': 1000,
                        'fixed_lag': 10,
                        'verbose': False})

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
                   'base_hessian': hessian_estimate,
                   'initial_params': (0.0, 0.1, 0.2),
                   'verbose': False
                   }
    mh = MetropolisHastings(sys_model, alg_type, mh_settings)

    if filter_method is 'kalman':
        if alg_type is 'mh0':
            mh.settings['step_size'] = 2.38 / np.sqrt(sys_model.no_params_to_estimate)
        elif alg_type is 'mh1':
            mh.settings['step_size'] = 1.38 / np.sqrt(sys_model.no_params_to_estimate**(1/3))
        elif alg_type is 'mh2':
            mh.settings['step_size'] = 0.8
        else:
            raise NameError("Unknown alg_type (mh0/mh1/mh2/qmh).")

        mh.run(kf)

    elif filter_method is 'particle':
        if alg_type is 'mh0':
            mh.settings['step_size'] = 2.562 / np.sqrt(sys_model.no_params_to_estimate)
        elif alg_type is 'mh1':
            mh.settings['step_size'] = 1.125 / np.sqrt(sys_model.no_params_to_estimate**(1/3))
        elif alg_type is 'mh2':
            mh.settings['step_size'] = 0.8
        else:
            raise NameError("Unknown alg_type (mh0/mh1/mh2/qmh).")
        mh.settings['step_size'] = 0.5 * mh.settings['step_size']
        mh.run(pf)
    else:
        raise NameError("Unknown filter_method (kalman/particle).")

    if plotting:
        mh.plot()
    else:
        mh.save_to_file(output_path='../results-tests',
                        sim_name='test_linear_gaussian_' + alg_type + '_' + filter_method,
                        sim_desc='...')


    print(np.cov(mh.params[500:2500,:], rowvar=False))