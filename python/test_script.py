# import tests.kalman_smoother as kalman_smoother
# kalman_smoother.run(cython_code=True)

# import tests.pf_linear_gaussian as pf
# pf.run(cython_code=True, save_to_file=True)

# import tests.pf_stochastic_volatility as pf
# pf.run(cython_code=True, save_to_file=False)

# import tests.mh_linear_gaussian as mh
# mh.run(cython_code=True, filter_method='kalman', alg_type='mh0', plotting=False)
# mh.run(cython_code=True, filter_method='particle', alg_type='mh0', plotting=False)
# mh.run(cython_code=True, filter_method='kalman', alg_type='mh1', plotting=False)
# mh.run(cython_code=True, filter_method='particle', alg_type='mh1', plotting=False)
# mh.run(cython_code=True, filter_method='kalman', alg_type='mh2', plotting=False)
# mh.run(cython_code=True, filter_method='particle', alg_type='mh2', plotting=False)

import tests.qmh_linear_gaussian as qmh
qmh.run(cython_code=True, filter_method='kalman', alg_type='bfgs', plotting=False)
# qmh.run(cython_code=True, filter_method='kalman', alg_type='sr1', plotting=False)
# qmh.run(cython_code=True, filter_method='particle', alg_type='bfgs', plotting=False)
# qmh.run(cython_code=True, filter_method='particle', alg_type='sr1', plotting=False)
# qmh.run(cython_code=True, filter_method='particle', alg_type='bfgs', plotting=False,
#         file_tag="useall", qn_only_accepted_info=False,
#         qn_memory_length=20)
# qmh.run(cython_code=True, filter_method='particle', alg_type='sr1', plotting=False,
#         file_tag="useall", qn_only_accepted_info=False,
#         qn_memory_length=20)
# qmh.run(cython_code=True, filter_method='particle', alg_type='bfgs', plotting=False,
#         file_tag="acceptall", qn_accept_all_initial=True)
# qmh.run(cython_code=True, filter_method='particle', alg_type='sr1', plotting=False,
#         file_tag="acceptall", qn_accept_all_initial=True)