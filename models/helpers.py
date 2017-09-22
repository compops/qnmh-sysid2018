import numpy as np
import copy

# Copy data from an instance of this struct to another
def getInferenceModel(oldModel, parametersToEstimate):
    newModel = copy.deepcopy(oldModel)
    newModel.modelType = "Inference model"
    newModel.noParametersToEstimate = len(parametersToEstimate)
    newModel.parametersToEstimate = parametersToEstimate
    
    newModel.parametersToEstimateIndex = []
    for parameter in newModel.parameters.keys():
        if parameter in parametersToEstimate:
            newModel.parametersToEstimateIndex.append(parametersToEstimate.index(parameter))
    return(newModel)


# Store the parameters into the struct
def template_storeParameters(inferenceModel, newParameters, completeModel):
    inferenceModel.par = np.zeros(completeModel.noParameters)

    for k in range(0, inferenceModel.noParametersToEstimate):
        inferenceModel.par[k] = np.array(newParameters[k], copy=True)

    for k in range(inferenceModel.noParametersToEstimate, completeModel.noParameters):
        inferenceModel.par[k] = completeModel.par[k]

# Returns the current parameters stored in this struct
def template_returnParameters(model):
    return(model.parameters[0:model.noParametersToEstimate])

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