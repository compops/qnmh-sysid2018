###############################################################################
#    Constructing Metropolis-Hastings proposals using damped BFGS updates
#    Copyright (C) 2018  Johan Dahlin < uni (at) johandahlin [dot] com >
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
###############################################################################

"""Helpers for resampling in the particle filter.
   The code is adopted from filterpy (https://github.com/rlabbe/filterpy)."""

from __future__ import absolute_import

import numpy as np
cimport numpy as np

DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t

DTYPE_float = np.float
ctypedef np.float_t DTYPE_float_t

def multinomial(np.ndarray[DTYPE_float_t] weights):
    """ Multinomial resampling.
        The code is adopted from filterpy (https://github.com/rlabbe/filterpy).
    """
    no_particles = len(weights)
    rnd_numbers = np.random.uniform(size=no_particles)

    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.
    return np.searchsorted(cumulative_sum, rnd_numbers)

def stratified(np.ndarray[DTYPE_float_t] weights):
    """ Stratified resampling
    The code is adopted from filterpy (https://github.com/rlabbe/filterpy).
    """
    no_particles = len(weights)
    rnd_numbers = np.random.uniform(size=no_particles)

    positions = (rnd_numbers + np.arange(no_particles)) / no_particles

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

def systematic(np.ndarray[DTYPE_float_t] weights):
    """ Systematic resampling
        The code is adopted from filterpy (https://github.com/rlabbe/filterpy).
    """
    no_particles = len(weights)
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
