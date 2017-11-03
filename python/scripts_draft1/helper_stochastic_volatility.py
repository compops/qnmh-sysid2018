import numpy as np

from models.stochastic_volatility_model import StochasticVolatilityModel
from state.particle_methods.standard import ParticleMethods
from parameter.mcmc.metropolis_hastings import MetropolisHastings


def run(mh_version, mh_settings, pf_settings, sim_name='test', sim_desc='',
        seed_offset=0):

    np.random.seed(87655678 + seed_offset)

    # System model
    sys_model = StochasticVolatilityModel()
    # sys_model.import_data_quandl(handle="NASDAQOMX/OMXS30",
    #                              start_date="2012-01-02",
    #                              end_date="2014-01-02",
    #                              variable='Index Value')

    sys_model.import_data_quandl(handle="BITSTAMP/USD",
                                 start_date="2014-04-15",
                                 end_date="2017-10-30",
                                 variable='VWAP')

    # Inference model
    sys_model.fix_true_params()
    sys_model.create_inference_model(params_to_estimate=('mu', 'phi', 'sigma_v'))

   # Particle filter and smoother
    pf = ParticleMethods(pf_settings)

    # Metropolis-Hastings
    mh = MetropolisHastings(sys_model, mh_version, mh_settings)
    mh.run(pf)

    mh.save_to_file(output_path='../results',
                    sim_name=sim_name,
                    sim_desc=sim_desc)

