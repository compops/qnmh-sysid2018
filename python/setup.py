import numpy
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(("state/kalman_methods/standard_filter.pyx",
                             "state/kalman_methods/rts_smoother.pyx",
                             "state/particle_methods/resampling.pyx",
                             "state/particle_methods/linear_gaussian_model.pyx")),
     include_dirs=[numpy.get_include()]
)