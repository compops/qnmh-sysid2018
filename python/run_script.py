import sys

import scripts_draft1.example1_lgss_kalman as example1
import scripts_draft1.example2_lgss_particles as example2
import scripts_draft1.example3_stochastic_volatility_particle as example3

if len(sys.argv) > 1:
    if int(sys.argv[1]) == 1:
        for i in range(10):
            example1.main(seed_offset=i)
    elif int(sys.argv[1]) == 2:
        for i in range(1, 10):
            example2.main(seed_offset=i)
    elif int(sys.argv[1]) == 3:
        example3.main(seed_offset=0)
    else:
        raise NameError("Unknown example to run...")

#example1.main(seed_offset=0)
#example2.main(seed_offset=0)
#example3.main(seed_offset=0)
