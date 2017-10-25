#import scripts.kalman_smoother as app
#import scripts.mh_zero_order as app
#import scripts.mh_first_order as app
#import scripts.mh_second_order as app
import scripts.qmh_sr1 as app

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

app.run()

