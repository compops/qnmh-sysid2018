import numpy as np
import copy

# Copy data from an instance of this struct to another
def getInferenceModel(oldModel, parametersToEstimate, unRestrictedParameters = True):
    newModel = copy.deepcopy(oldModel)
    newModel.modelType = "Inference model"
    if unRestrictedParameters:
        newModel.parameterisation = "unrestricted"
    if isinstance(parametersToEstimate, str):
        newModel.noParametersToEstimate = 1
    else:
        newModel.noParametersToEstimate = len(parametersToEstimate)
    newModel.parametersToEstimate = parametersToEstimate
    newModel.trueParameters = oldModel.parameters
    
    newModel.parametersToEstimateIndex = []
    for param in newModel.parameters.keys():
        if param in parametersToEstimate:
            newModel.parametersToEstimateIndex.append(parametersToEstimate.index(param))
    return(newModel)

# Store the parameters into the struct
def template_storeParameters(model, newParameters):
    model.parameters = model.trueParameters
    if isinstance(newParameters, float) or (isinstance(newParameters, np.ndarray) and len(newParameters) == 1):
        model.parameters[model.parametersToEstimate] = newParameters
    else:
        for param in model.parametersToEstimate:
            model.parameters[param] = newParameters[model.parametersToEstimate.index(param)]

# Store the parameters into the struct
def template_getParameters(model):
    parameters = []
    if isinstance(model.parametersToEstimate, str):
        parameters.append(model.parameters[model.parametersToEstimate])
    else:
        for param in model.parametersToEstimate:
            parameters.append(model.parameters[param])
    return np.array(parameters)
    
# Standard template for importing data
def template_importData(model, fileName):
    data = np.loadtxt(fileName, delimiter=",")
    model.observations = np.array(data[0:model.noObservations], copy=True).reshape((model.noObservations, 1))

# Standard template for generating data
def template_generateData(model):
    model.states = np.zeros((model.noObservations + 1, 1))
    model.observations = np.zeros((model.noObservations, 1))
    model.states[0] = model.initialState
    for t in range(0, model.noObservations):
        model.observations[t] = model.generateObservation(model.states[t])
        model.states[t + 1] = model.generateState(model.states[t])
    model.states = model.states[0:model.noObservations]