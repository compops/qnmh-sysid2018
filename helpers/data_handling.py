"""Helpers for generating and importing data from/to models."""
import pandas as pd
import numpy as np
import quandl

def import_data_quandl(model, handle, start_date, end_date, variable):
    """Imports financial data from Quandl and computes the log-returns
    for the use in stochastic volatility models."""
    data = quandl.get(handle, start_date=start_date, end_date=end_date)
    log_returns = 100 * np.diff(np.log(data[variable]))
    model.no_obs = len(log_returns) - 1
    obs = np.array(log_returns, copy=True).reshape((model.no_obs + 1, 1))
    model.obs = obs

def import_data(model, file_name):
    """Imports data from file_name to model. Data is given as a csv file with
    headers observation and state. Note that only observation are required."""
    data_frame = pd.read_csv(file_name)

    if 'observation' in list(data_frame):
        obs = data_frame['observation'].values[0:(model.no_obs + 1)]
        obs = np.array(obs, copy=True).reshape((model.no_obs + 1, 1))
        model.obs = obs

    if 'state' in list(data_frame):
        states = data_frame['state'].values[0:(model.no_obs + 1)]
        states = np.array(states, copy=True).reshape((model.no_obs + 1, 1))
        model.states = states

    if 'input' in list(data_frame):
        inputs = data_frame['input'].values[0:(model.no_obs + 1)]
        inputs = np.array(obs, copy=True).reshape((model.no_obs + 1, 1))
        model.inputs = inputs

    else:
        model.states = None

    print("Loaded data from file: " + file_name + ".")

def generate_data(model, file_name=None):
    """Generated data from model and saves the data to a csv file with name
    file_name if required."""
    model.states = np.zeros((model.no_obs + 1, 1))
    model.obs = np.zeros((model.no_obs + 1, 1))
    model.states[0] = model.initial_state

    for i in range(1, model.no_obs + 1):
        model.states[i] = model.generate_state(model.states[i-1], i)
        model.obs[i] = model.generate_obs(model.states[i], i)

    if file_name:
        data_frame = pd.DataFrame(data=np.hstack((model.states, model.obs)),
                                  columns=['state', 'observation'])
        data_frame.to_csv(file_name, index=False, header=True)
        print("Wrote generated data to file: " + file_name + ".")
