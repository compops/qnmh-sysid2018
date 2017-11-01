"""Helpers for manipulating the file system."""
import json
import os

import numpy as np

def ensure_dir(file_name):
    """ Check if dirs for outputs exists, otherwise create them"""
    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

def write_to_json(data, output_path, sim_name, output_type):
    """Writes results in the form of a dictionary to file as JSON."""
    # Convert NumPy arrays to lists
    for key in data:
        if isinstance(data[key], np.ndarray):
            data[key] = data[key].tolist()

    # Check if the directories exists and write data as json
    file_name = output_path + '/' + sim_name + '/' + output_type
    ensure_dir(file_name)

    with open(file_name, 'w') as f:
        json.dump(data, f, ensure_ascii=False)

    print("Wrote results to: " + file_name + ".")