#import scripts.kalmanSmoother as kalmanSmoother
#kalmanSmoother.run()

#import scripts.zeroOrderMetropolisHastings as zeroOrderMetropolisHastings
#zeroOrderMetropolisHastings.run()

#import scripts.firstOrderMetropolisHastings as firstOrderMetropolisHastings
#firstOrderMetropolisHastings.run()

#import scripts.secondOrderMetropolisHastings as secondOrderMetropolisHastings
#secondOrderMetropolisHastings.run()

import scripts.quasiNewtonMetropolisHastings as quasiNewtonMetropolisHastings
mhSampler = quasiNewtonMetropolisHastings.run()