import numpy as np
import matplotlib.pylab as plt
from state import kalman
from models import lgss_4parameters
from models.helpers import getInferenceModel

# System model
systemModel = lgss_4parameters.model()
systemModel.parameters['mu'] = 0.2
systemModel.parameters['phi'] = 0.98
systemModel.parameters['sigma_v'] = 0.16
systemModel.parameters['sigma_e'] = 0.1
systemModel.noObservations = 5000
systemModel.initialState = 0.0
systemModel.generateData()

# Inference model
inferenceModel = getInferenceModel(systemModel, parametersToEstimate = ('mu', 'phi'))

# Kalman filter and smoother
km = kalman.kalmanMethods()
km.smoother(inferenceModel)
#plt.plot(np.arange(systemModel.noObservations), systemModel.states, np.arange(systemModel.noObservations), km.filteredStateEstimate)
#plt.show()

grid_mu = np.arange(-1, 1, 0.01)
gradient_mu = np.zeros(len(grid_mu))
for i in range(len(grid_mu)):
    inferenceModel.parameters['mu'] = grid_mu[i]
    km.smoother(inferenceModel)
    gradient_mu[i] = km.gradient[0]

plt.plot(grid_mu, gradient_mu)
plt.show()

