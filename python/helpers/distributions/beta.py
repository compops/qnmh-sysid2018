"""Helpers for the Beta distribtion."""
import numpy as np
import scipy as sp

def pdf(param, alpha, beta):
    """ Computes the pdf of the Beta distribution.

        Args:
            param: value to evaluate in
            alpha: shape parameter
            beta: shape parameter

        Returns:
            A scalar with the value of the pdf.

    """
    part1 = sp.special.gamma(alpha + beta)
    part2 = sp.special.gamma(alpha) + sp.special.gamma(beta)
    part3 = param**(alpha - 1.0) * (1.0 - param)**(beta - 1.0)
    return part1 / part2 * part3

def logpdf(param, alpha, beta):
    """ Computes the log-pdf of the Beta distribution.

        Args:
            param: value to evaluate in
            alpha: shape parameter
            beta: shape parameter

        Returns:
            A scalar with the value of the log-pdf.

    """
    part1 = sp.special.gammaln(alpha + beta)
    part2 = -np.log(sp.special.gamma(alpha) + sp.special.gamma(beta))
    part3 = (alpha - 1.0) * np.log(param) * (beta - 1.0) * np.log(1.0 - param)
    return  part1 + part2 + part3

def logpdf_gradient(param, alpha, beta):
    """ Computes the gradient of the log-pdf of the Beta distribution.

        Args:
            param: value to evaluate in
            alpha: shape parameter
            beta: shape parameter

        Returns:
            A scalar with the value of the gradient of the log-pdf.

    """
    return (alpha - 1.0) / param + (1.0 - beta) / (1.0 - param)

def logpdf_hessian(param, alpha, beta):
    """Computes the Hessian of the log-pdf of the Beta distribution.

    Args:
        param: value to evaluate in
        alpha: shape parameter
        beta: shape parameter

    Returns:
        A scalar with the value of the Hessian of the log-pdf.
    """
    return -(alpha - 1.0) / (param**2) + (1.0 - beta) / ((1.0 - param)**2)