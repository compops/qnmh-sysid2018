import numpy as np

def computeIntegratedAutoCorrelationTime(sampler, parameterTrace=False, maxLag=False):

    def estimateIACT(x, m, maxLag):
        # code from http://stackoverflow.com/questions/14297012/estimate-autocorrelation-using-python
        n = len(x)
        d = np.asarray(x)
        nmax = int(np.floor(n / 10.0))
        c0 = np.sum((x - m) ** 2) / float(n)
        def r(h):
            acf_lag = ((d[:n - h] - m) * (d[h:] - m)).sum() / float(n) / c0
            return round(acf_lag, 3)
        x = np.arange(n)
        acf_coeffs = map(r, x)
        if (maxLag == False):
            try:
                cutoff = np.where(
                    np.abs(acf_coeffs[0:int(nmax)]) < 2.0 / np.sqrt(n))[0][0]
            except:
                cutoff = nmax
        else:
            cutoff = maxLag
        tmp = int(min(cutoff, nmax))
        return 1.0 + 2.0 * np.sum(acf_coeffs[0:tmp])
    
    IACT = np.zeros(sampler.nPars)
    MCtrace = sampler.th[range(int(sampler.noBurnInIter), int(sampler.iter)), :]
    if (parameterTrace == False):
        parameterTrace = np.mean(MCtrace, axis=0)
    for i in range(sampler.nPars):
        IACT[i] = estimateIACT(MCtrace[:, i], parameterTrace[i], maxLag)
    return IACT

def computeEffectiveSampleSize(sampler, parameterTrace=False, maxLag=False):
    return((sampler.nIter - sampler.noBurnInIter) / computeIntegratedAutoCorrelationTime(sampler, parameterTrace, maxLag))

def computeSquaredJumpDistance(sampler):
    MCtrace = sampler.th[range(int(sampler.noBurnInIter), int(sampler.iter)), :]
    return np.sum(np.linalg.norm(np.diff(MCtrace, axis=0), 2, axis=1)**2) / (MCtrace.shape[0] - 1.0)