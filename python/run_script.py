from joblib import Parallel, delayed
import multiprocessing

import scripts_draft1.example1_lgss_kalman as example1
import scripts_draft1.example2_lgss_particles as example2
import scripts_draft1.example3_stochastic_volatility_particle as example3

#num_cores = multiprocessing.cpu_count()
num_cores = 4
count = range(10)

#res1 = Parallel(n_jobs=num_cores)(delayed(example1.main)(i) for i in count)
#res2 = Parallel(n_jobs=num_cores)(delayed(example2.main)(i) for i in count)
# example3.run(seed_offset=0)

# for i in range(10):
#     example1n.main(seed_offset=i)

# for i in range(10):
#     example2.main(seed_offset=i)

# example3.main(seed_offset=0)

#example1.main(seed_offset=)
example2.main(seed_offset=0)
#example3.main(seed_offset=0)
