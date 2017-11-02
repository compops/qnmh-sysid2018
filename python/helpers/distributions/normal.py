"""Helpers for the Gaussian distribtion."""

import numpy as np

def pdf(parm, mean, stdev):
    """ Computes the pdf of the Gaussian distribution.

        Args:
            parm: value to evaluate in
            mean: mean
            stdev: standard deviation

        Returns:
            A scalar with the value of the pdf.

    """
    quad_term = -0.5 / (stdev**2) * (parm - mean)**2
    return 1.0 / np.sqrt(2 * np.pi * stdev**2) * np.exp(quad_term)

def logpdf(parm, mean, stdev):
    """ Computes the log-pdf of the Gaussian distribution.

        Args:
            parm: value to evaluate in
            mean: mean
            stdev: standard deviation

        Returns:
            A scalar with the value of the log-pdf.

    """
    quad_term = -0.5 / (stdev**2) * (parm - mean)**2
    return -0.5 * np.log(2 * np.pi * stdev**2) + quad_term

def logpdf_gradient(parm, mean, stdev):
    """ Computes the gradient of the log-pdf of the Gaussian distribution.

        Args:
            parm: value to evaluate in
            mean: mean
            stdev: standard deviation

        Returns:
            A scalar with the value of the gradient of the log-pdf.

    """
    return -(mean - parm) / stdev**2

def logpdf_hessian(parm, mean, stdev):
    """ Computes the Hessian of the log-pdf of the Gaussian distribution.

        Args:
            parm: value to evaluate in
            mean: mean
            stdev: standard deviation

        Returns:
            A scalar with the value of the Hessian of the log-pdf.

    """
    return -1.0 / stdev**2
