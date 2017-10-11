
import numpy as np
identityMatrix = np.diag(np.ones(3))
inverseHessianEstimate = 0.01 * identityMatrix
noEffectiveSamples = 0

parametersDiff = np.array((0.1, 0.2, 0.3))
gradientsDiff = np.array((2.0, 3.0, 4.0))

B = np.matmul(inverseHessianEstimate, inverseHessianEstimate.transpose())

term1 = np.dot(parametersDiff, gradientsDiff)
term2 = np.dot(np.dot(parametersDiff, B), parametersDiff)

if term1 > 0.2 * term2:
    theta = 1.0
else:
    theta = 0.5 * term2 / (term2 - term1)

r = theta * gradientsDiff + (1.0 - theta) * np.dot(B, parametersDiff)

quadraticFormSB = np.dot(np.dot(parametersDiff, B), parametersDiff)
t = parametersDiff / quadraticFormSB

u1 = np.sqrt(quadraticFormSB / np.dot(parametersDiff, r))

u2 = np.dot(B, parametersDiff)

u = u1 * r + u2

inverseHessianEstimate = np.matmul(identityMatrix - np.outer(u, t), inverseHessianEstimate)            
noEffectiveSamples += 1

mh.noEffectiveSamples[mh.currentIteration] = noEffectiveSamples

hessianEstimate = np.linalg.inv(inverseHessianEstimate)
hessianEstimate = np.matmul(hessianEstimate, hessianEstimate.transpose())
#print(np.linalg.eig(hessianEstimate)[0])
#print(hessianEstimate)
return hessianEstimate