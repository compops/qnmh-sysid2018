import numpy as np

from models.linear_gaussian_model import LinearGaussianModel
from state.kalman_methods.standard import KalmanMethods
from state.particle_methods.standard import ParticleMethods
from parameter.mcmc.metropolis_hastings import MetropolisHastings

def run(mh_version, mh_settings, kf_settings, pf_settings,
        smoothing_method="kalman", sim_name='test', sim_desc='', seed_offset=0):

    np.random.seed(87655678 + seed_offset)

    # System model
    sys_model = LinearGaussianModel()
    sys_model.import_data(file_name="../data/linear_gaussian_model/linear_gaussian_model_T1000_goodSNR.csv")

    # Inference model
    sys_model.fix_true_params()
    sys_model.create_inference_model(params_to_estimate=('mu', 'phi', 'sigma_v'))

    # Kalman filter and smoother
    kf = KalmanMethods(kf_settings)

    # Particle filter and smoother
    pf = ParticleMethods(pf_settings)

    # Metropolis-Hastings
    mh = MetropolisHastings(sys_model, mh_version, mh_settings)

    if smoothing_method is 'kalman':
        mh.run(kf)
    elif smoothing_method is 'particle':
        mh.run(pf)
    else:
        raise NameError("Unknown smoothing method selected")

    mh.save_to_file(output_path='../results',
                    sim_name=sim_name,
                    sim_desc=sim_desc)
