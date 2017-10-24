"""Helpers for manipulating the file system."""

import os

def ensure_dir(file_name):
    """ Check if dirs for outputs exists, otherwise create them"""
    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
