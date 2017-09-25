import numpy as np
import copy

# Copy data from an instance of this struct to another
def getInferenceModel(oldModel, parametersToEstimate):
    newModel = copy.deepcopy(oldModel)
    newModel.modelType = "Inference model"
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
    if isinstance(newParameters, float):
        for param in model.parametersToEstimate:
            model.parameters[param] = newParameters
    else:
        for param in model.parametersToEstimate:
            model.parameters[param] = newParameters[model.parametersToEstimate.index(param)]

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