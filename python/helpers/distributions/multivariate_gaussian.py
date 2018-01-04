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

"""Helpers for the multivariate Gaussian distribution."""
import numpy as np

def logpdf(parm, mean, cov_matrix):
    """ Computes the log-pdf of the multivariate Gaussian distribution.

        Args:
            parm: value to evaluate in
            mean: mean vector
            cov_matrix: covariance matrix

        Returns:
            A scalar with the value of the pdf.

    """
    no_dimensions = len(cov_matrix)

    norm_coeff = no_dimensions * np.log(2.0 * np.pi)
    norm_coeff += np.linalg.slogdet(cov_matrix)[1]
    error = parm - mean

    quad_term = np.dot(error, np.linalg.pinv(cov_matrix))
    quad_term = np.dot(quad_term, error.transpose())
    return -0.5 * (norm_coeff + quad_term)
