import distutils.sysconfig
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
setup(
cmdclass = {'build_ext': build_ext},
ext_modules = [
Extension("my_linregress",
["my_linregress.pyx"],
include_dirs=[numpy.get_include()],
libraries=["m"],
),
])
