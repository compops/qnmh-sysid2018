"""Helpers for resampling in the particle filter.
   The code is adopted from filterpy (https://github.com/rlabbe/filterpy)."""
import numpy as np

def multinomial(weights, rnd_numbers=None):
    """ Multinomial resampling.
        The code is adopted from filterpy (https://github.com/rlabbe/filterpy).
    """
    no_particles = len(weights)
    if rnd_numbers is None:
        rnd_numbers = np.random.uniform(size=no_particles)

    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.
    return np.searchsorted(cumulative_sum, rnd_numbers)

def stratified(weights, rnd_numbers=None):
    """ Stratified resampling
    The code is adopted from filterpy (https://github.com/rlabbe/filterpy).
    """
    no_particles = len(weights)
    if rnd_numbers is None:
        rnd_numbers = np.random.uniform(size=no_particles)

    positions = (rnd_numbers + range(no_particles)) / no_particles

    indexes = np.zeros(no_particles, 'i')
    cumulative_sum = np.cumsum(weights)
    i = 0
    j = 0
    while i < no_particles:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes

def systematic(weights, rnd_number=None):
    """ Systematic resampling
        The code is adopted from filterpy (https://github.com/rlabbe/filterpy).
    """
    no_particles = len(weights)
    if rnd_number is None:
        rnd_number = np.random.uniform()

    positions = (np.arange(no_particles) + rnd_number) / no_particles

    indexes = np.zeros(no_particles, 'i')
    cumulative_sum = np.cumsum(weights)
    i = 0
    j = 0
    while i < no_particles:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1

    return indexes
