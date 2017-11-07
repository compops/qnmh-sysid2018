"""Helpers for manipulating the file system."""
import json
import gzip
import os

import numpy as np

def ensure_dir(file_name):
    """ Check if dirs for outputs exists, otherwise create them

        Args:
            file_name: relative search path to file (string).

        Returns:
           Nothing.

    """
    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

def write_to_json(data, output_path, sim_name, output_type, as_gzip=True):
    """ Writes result of state/parameter estimation to file.

        Writes results in the form of a dictionary to file as JSON.

        Args:
            data: dict to store to file.
            output_path: relative file path to dir to store file in.
                         Without / at the end.
            sim_name: name of simulation (determines search path).
                      Without / at the end.
            output_type: name of the type of output (determines file name)

        Returns:
           Nothing.

    """
    # Convert NumPy arrays to lists
    for key in data:
        if isinstance(data[key], np.ndarray):
            data[key] = data[key].tolist()

    # Check if the directories exists and write data as json
    file_name = output_path + '/' + sim_name + '/' + output_type
    ensure_dir(file_name)

    if as_gzip:
        with gzip.GzipFile(file_name + '.gz', 'w') as fout:
            json_str = json.dumps(data)
            json_bytes = json_str.encode('utf-8')
            fout.write(json_bytes)
    else:
        with open(file_name, 'w') as f:
            json.dump(data, f, ensure_ascii=False)

    print("Wrote results to: " + file_name + ".")
