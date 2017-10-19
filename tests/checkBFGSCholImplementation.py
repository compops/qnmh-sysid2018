import json
import numpy as np

###############################################################################
# Load data
###############################################################################

with open('tests/dataFromRun.json') as data_file:    
    data = json.load(data_file)

allParameters = np.array(data['parameters'])
allGradients = np.array(data['gradient'])
allHessians = np.array(data['hessian'])
allAccepted = np.array(data['accepted'])
allLogPrior = np.array(data['logPrior'])
allLogLikelihood = np.array(data['logLikelihood'])

###############################################################################
# Setup BFGS update
###############################################################################
currentIteration = 20
memoryLength = 10
baseStepSize = 0.01
initialHessian = 0.01

noParameters = allParameters.shape[1]
identityMatrix = np.diag(np.ones(noParameters))

# Extract parameters and gradidents
idx = range(currentIteration - memoryLength, currentIteration)
parameters = allParameters[idx, :]
gradients = allGradients[idx, :]
hessians = allHessians[idx, :, :]
accepted = allAccepted[idx]
target = allLogPrior[idx] + allLogLikelihood[idx]

# Keep only unique parameters and gradients
idx = np.where(accepted > 0)[0]
parameters = parameters[idx, :]
gradients = gradients[idx, :]
hessians = hessians[idx, :, :]
target = np.concatenate(target[idx]).reshape(-1)    
accepted = accepted[idx, :]

# Sort and compute differences
idx = np.argsort(target)
parameters = parameters[idx, :]
gradients = gradients[idx, :]
hessians = hessians[idx, :]

parametersDiff = np.zeros((len(idx) - 1, noParameters))
gradientsDiff = np.zeros((len(idx) - 1, noParameters))

for i in range(len(idx) - 1):
    parametersDiff[i, :] = parameters[i+1, :] - parameters[i, :]
    gradientsDiff[i, :] = gradients[i+1, :] - gradients[i, :]
    print(np.dot(parametersDiff[i], gradientsDiff[i]))

inverseHessianEstimate = initialHessian * identityMatrix
noEffectiveSamples = 0

for i in range(parametersDiff.shape[0]):
    B = np.matmul(inverseHessianEstimate, inverseHessianEstimate.transpose())
    doUpdate = False

    if True:
        term1 = np.dot(parametersDiff[i], gradientsDiff[i])
        term2 = np.dot(np.dot(parametersDiff[i], B), parametersDiff[i])

        if (term1 > 0.2 * term2):
            theta = 1.0
        else:
            theta = 0.8 * term2 / (term2 - term1)
        
        r = theta * gradientsDiff[i] + (1.0 - theta) * np.dot(B, parametersDiff[i])
        doUpdate = True
    else:
        if np.dot(parametersDiff[i], gradientsDiff[i]) > 0:
            doUpdate = True
            r = gradientsDiff[i]

    if doUpdate:
        quadraticFormSB = np.dot(np.dot(parametersDiff[i], B), parametersDiff[i])
        t = parametersDiff[i] / quadraticFormSB
        
        u1 = np.sqrt(quadraticFormSB / np.dot(parametersDiff[i], r))
        u2 = np.dot(B, parametersDiff[i])
        u = u1 * r + u2

        inverseHessianEstimate = np.matmul(identityMatrix - np.outer(u, t), inverseHessianEstimate)            
        noEffectiveSamples += 1

mh.noEffectiveSamples[currentIteration] = noEffectiveSamples

inverseHessianEstimateSquared = np.dot(inverseHessianEstimate, inverseHessianEstimate.transpose())
naturalGradient = np.dot(inverseHessianEstimateSquared, allGradients[currentIteration, :])
return inverseHessianEstimate, naturalGradient