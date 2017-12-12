from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

# setup(
#     ext_modules = cythonize("helloworld.pyx"),
#     include_dirs=[numpy.get_include()]
# )

setup(
    ext_modules=[
        Extension("helloworld", ["helloworld.c"],
                  include_dirs=[numpy.get_include()]),
    ],
)
