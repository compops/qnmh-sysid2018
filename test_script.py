#import tests.kalman_smoother as kalman_smoother
#kalman_smoother.run()


#import tests.particle_smoother_linear_gaussian as particle_smoother
import tests.particle_smoother_phillips as particle_smoother
particle_smoother.run(cython_code=False)

# import tests.mh_zero_order as mh_zero_order
# mh_zero_order.run()

# import tests.mh_first_order as mh_first_order
# mh_first_order.run()

# import tests.mh_second_order as mh_second_order
# mh_second_order.run()

# import tests.qmh_sr1 as qmh_sr1
# qmh_sr1.run()

# import tests.qmh_bfgs as qmh_bfgs
# qmh_bfgs.run()
