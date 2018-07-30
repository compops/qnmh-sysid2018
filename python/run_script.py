import sys

import scripts.example1_lgss_kalman as example1
import scripts.example2_lgss_particles as example2
import scripts.example3_stochastic_volatility as example3

if len(sys.argv) > 1:
    if (len(sys.argv) > 2) and int(sys.argv[2]) == 1:
        print("Running full experiment (25 Monte Carlo runs).")
        print("This will probably take a few hours.")
        NO_ITERS = 25
    else:
        print("Running reduced experiment (1 Monte Carlo run).")
        NO_ITERS = 1

    if int(sys.argv[1]) == 1:
        print("Running first example.")
        for i in range(NO_ITERS):
            example1.main(seed_offset=i)

    elif int(sys.argv[1]) == 2:
        print("Running second example.")
        for i in range(NO_ITERS):
            example2.main(seed_offset=i)

    elif int(sys.argv[1]) == 3:
        print("Running third example.")
        example3.main(seed_offset=0)

    else:
        raise NameError("Unknown example.")
else:
    raise NameError("Need to provide the experiment to run.")
