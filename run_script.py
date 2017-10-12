#import scripts.kalmanSmoother as kalmanSmoother
#kalmanSmoother.run()

# import scripts.zeroOrderMetropolisHastings as zeroOrderMetropolisHastings
# zeroOrderMetropolisHastings.run()

# import scripts.firstOrderMetropolisHastings as firstOrderMetropolisHastings
# firstOrderMetropolisHastings.run()

# import scripts.secondOrderMetropolisHastings as secondOrderMetropolisHastings
# mhSampler = secondOrderMetropolisHastings.run()

import scripts.quasiNewtonMetropolisHastings as quasiNewtonMetropolisHastings
mhSampler = quasiNewtonMetropolisHastings.run()

# import json
# output = {}
# output.update({'parameters': mhSampler.parameters[900:, :].tolist()})
# output.update({'gradient': mhSampler.gradient[900:, :].tolist()})
# output.update({'hessian': mhSampler.hessian[900:, :, :].tolist()})
# output.update({'accepted': mhSampler.accepted[900:].tolist()})
# output.update({'logPrior': mhSampler.logPrior[900:].tolist()})
# output.update({'logLikelihood': mhSampler.logLikelihood[900:].tolist()})
# with open('tests/dataFromRun.json', 'w') as f:
#     json.dump(output, f, ensure_ascii=False)