# import tests.kalman_smoother as kalman_smoother
# kalman_smoother.run()

#import tests.pf_linear_gaussian as pf
#import tests.pf_stochastic_volatility as pf
#pf.run(cython_code=False, save_to_file=True)

import tests.mh_linear_gaussian as mh_linear_gaussian
# mh_linear_gaussian.run(filter_method='kalman', alg_type='mh0', plotting=False)
# mh_linear_gaussian.run(filter_method='particle', alg_type='mh0', plotting=False)
mh_linear_gaussian.run(filter_method='kalman', alg_type='mh1', plotting=False)
mh_linear_gaussian.run(filter_method='particle', alg_type='mh1', plotting=False)
mh_linear_gaussian.run(filter_method='kalman', alg_type='mh2', plotting=False)
mh_linear_gaussian.run(filter_method='particle', alg_type='mh2', plotting=False)

# import tests.mh_first_order as mh_first_order
# mh_first_order.run()

# import tests.mh_second_order as mh_second_order
# mh_second_order.run()

# import tests.qmh_sr1 as qmh_sr1
# qmh_sr1.run()

# import tests.qmh_bfgs as qmh_bfgs
# qmh_bfgs.run()
