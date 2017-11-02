"""Helpers for the multivariate Gaussian distribtion."""
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
