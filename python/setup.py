import numpy
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(("state/kalman_methods/cython_helper.pyx",
                             "state/particle_methods/resampling.pyx",
                             "state/particle_methods/cython_lgss_helper.pyx",
                             "state/particle_methods/cython_sv_helper.pyx",
                             "state/particle_methods/cython_sv_leverage_helper.pyx"
                             )),
                             #gdb_debug=True),
     include_dirs=[numpy.get_include()]
)