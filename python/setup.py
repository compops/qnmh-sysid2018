import numpy
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(("helpers/cython/rnd_numbers.pyx",
                             "state/kalman_methods/cython_helper.pyx",
                             "state/particle_methods/resampling.pyx",
                             "state/particle_methods/cython_helper.pyx")),
     include_dirs=[numpy.get_include()]
)