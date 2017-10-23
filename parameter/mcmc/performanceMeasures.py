import numpy as np
import matplotlib.pylab as plt

def calcIACT(sampler, maxLag=None):

    def helperIACT(x, maxLag):
        n = len(x)
        variance = np.var(x)
        x = x - np.mean(x)
        r = np.correlate(x, x, mode = 'full')[-n:]
        result = r / (variance * (np.arange(n, 0, -1)))
        
        if not maxLag:
            maxLag = np.where(np.abs(result) < 1.96 / np.sqrt(n))[0][0]
        return 1.0 + 2.0 * np.sum(result[0:maxLag])
    
    IACT = np.zeros(sampler.noParametersToEstimate)
    MCtrace = sampler.unrestrictedParameters[range(int(sampler.settings['noBurnInIters']), int(sampler.currentIteration)), :]
    parameterTrace = np.mean(MCtrace, axis=0)
    
    for i in range(sampler.noParametersToEstimate):
        IACT[i] = helperIACT(MCtrace[:, i], maxLag)
    return IACT

def calcESS(sampler, maxLag=None):
    return((sampler.currentIteration - sampler.settings['noBurnInIters']) / calcIACT(sampler, maxLag))

def calcSJD(sampler):
    MCtrace = sampler.unrestrictedParameters[range(int(sampler.settings['noBurnInIters']), int(sampler.currentIteration)), :]
    return np.sum(np.linalg.norm(np.diff(MCtrace, axis=0), 2, axis=1)**2) / (MCtrace.shape[0] - 1.0)