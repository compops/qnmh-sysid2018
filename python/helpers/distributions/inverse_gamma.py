"""Helpers for the inverse Gamma distribtion."""
import numpy as np
import scipy as sp

def pdf(param, shape, rate):
    """ Computes the pdf of the inverse Gamma distribution.

        Args:
            param: value to evaluate in
            shape: shape parameter
            rate: rate parameter

        Returns:
            A scalar with the value of the pdf.

    """
    coef = rate**shape / sp.special.gamma(shape)
    return coef * param**(-shape - 1.0) * np.exp(-rate / param)

def logpdf(param, shape, rate):
    """ Computes the log-pdf of the inverse Gamma distribution.

        Args:
            param: value to evaluate in
            shape: shape parameter
            rate: rate parameter

        Returns:
            A scalar with the value of the log-pdf.

    """
    part1 = shape * np.log(rate) - sp.special.gammaln(shape)
    part2 = (-shape - 1.0) * np.log(param) - (rate / x)
    return part1 + part2

def logpdf_gradient(param, shape, rate):
    """ Computes the gradient of the log-pdf of the inverse Gamma distribution.

        Args:
            param: value to evaluate in
            shape: shape parameter
            rate: rate parameter

        Returns:
            A scalar with the value of the gradient of the log-pdf.

    """
    return (-shape - 1.0) / param + rate / (param**2)

def logpdf_hessian(param, shape, rate):
    """ Computes the Hessian of the log-pdf of the inverse Gamma distribution.

        Args:
            param: value to evaluate in
            shape: shape parameter
            rate: rate parameter

        Returns:
            A scalar with the value of the Hessian of the log-pdf.

    """
    return -(-shape - 1.0) / (param**2) - 2.0 * rate / (param**3)