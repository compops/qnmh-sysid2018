import pandas as pd
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
    newModel.trueParameters = copy.deepcopy(oldModel.parameters)
    
    newModel.parametersToEstimateIndex = []
    for param in newModel.parameters.keys():
        if param in parametersToEstimate:
            newModel.parametersToEstimateIndex.append(parametersToEstimate.index(param))
    return(newModel)

# Store unrestricted parameters into the struct and update the restricted
def template_storeUnrestrictedParameters(model, newParameters):
    model.parameters = copy.deepcopy(model.trueParameters)
    model.transformParametersToUnrestricted()
    if isinstance(newParameters, float) or (isinstance(newParameters, np.ndarray) and len(newParameters) == 1):
        model.unrestrictedParameters[model.parametersToEstimate] = float(newParameters)
    else:
        for param in model.parametersToEstimate:
            model.unrestrictedParameters[param] = float(newParameters[model.parametersToEstimate.index(param)])
    model.transformParametersFromUnrestricted()

# Store restricted parameters into the struct and update the unrestricted
def template_storeRestrictedParameters(model, newParameters):
    model.parameters = copy.deepcopy(model.trueParameters)
    if isinstance(newParameters, float) or (isinstance(newParameters, np.ndarray) and len(newParameters) == 1):
        model.parameters[model.parametersToEstimate] = float(newParameters)
    else:
        for param in model.parametersToEstimate:
            model.parameters[param] = float(newParameters[model.parametersToEstimate.index(param)])
    model.transformParametersToUnrestricted()

# Get the unrestricted parameters from the struct
def template_getUnrestrictedParameters(model):
    parameters = []
    if isinstance(model.parametersToEstimate, str):
        parameters.append(model.unrestrictedParameters[model.parametersToEstimate])
    else:
        for param in model.parametersToEstimate:
            parameters.append(model.unrestrictedParameters[param])
    return np.array(parameters)

# Get the Restricted parameters from the struct
def template_getRestrictedParameters(model):
    parameters = []
    if isinstance(model.parametersToEstimate, str):
        parameters.append(model.parameters[model.parametersToEstimate])
    else:
        for param in model.parametersToEstimate:
            parameters.append(model.parameters[param])
    return np.array(parameters)

# Standard template for importing data
def template_importData(model, fileName):
    df = pd.read_csv(fileName)

    if 'observation' in list(df):
        observations = df['observation'].values[1:(model.noObservations+1)]
        observations = np.array(observations, copy=True).reshape((model.noObservations, 1))
        model.observations = observations
    
    if 'state' in list(df):
        states = df['state'].values[0:(model.noObservations+1)]
        states = np.array(states, copy=True).reshape((model.noObservations+1, 1))
        model.states = states        
    
    print("Loaded data from file: " + fileName + ".")
    #data = np.loadtxt(fileName, delimiter=",")
    #model.observations = np.array(data[0:model.noObservations], copy=True).reshape((model.noObservations, 1))

# Standard template for generating data
def template_generateData(model, fileName=None):
    model.states = np.zeros((model.noObservations + 1, 1))
    model.observations = np.zeros((model.noObservations + 1, 1))
    model.states[0] = model.initialState
    for t in range(1, model.noObservations + 1):
        model.states[t] = model.generateState(model.states[t-1])
        model.observations[t] = model.generateObservation(model.states[t])
    
    if fileName:
        df = pd.DataFrame(data=np.hstack((model.states, model.observations)), columns=['state', 'observation'])
        df.to_csv(fileName, index=False, header=True)
        print("Wrote generated data to file: " + fileName + ".")