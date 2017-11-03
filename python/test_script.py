# import tests.kalman_smoother as kalman_smoother
# kalman_smoother.run()

# import tests.pf_linear_gaussian as pf
# pf.run(cython_code=False, save_to_file=True)

# import tests.pf_stochastic_volatility as pf
# pf.run(cython_code=False, save_to_file=True)

import tests.mh_linear_gaussian as mh
mh.run(filter_method='kalman', alg_type='mh0', plotting=False)
# mh.run(filter_method='particle', alg_type='mh0', plotting=False)
# mh.run(filter_method='kalman', alg_type='mh1', plotting=False)
# mh.run(filter_method='particle', alg_type='mh1', plotting=False)
# mh.run(filter_method='kalman', alg_type='mh2', plotting=False)
# mh.run(filter_method='particle', alg_type='mh2', plotting=False)

# import tests.qmh_linear_gaussian as qmh
# qmh.run(filter_method='kalman', alg_type='bfgs', plotting=False)
# qmh.run(filter_method='kalman', alg_type='sr1', plotting=False)
# qmh.run(filter_method='particle', alg_type='bfgs', plotting=False)
# qmh.run(filter_method='particle', alg_type='sr1', plotting=False)