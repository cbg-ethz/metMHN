from setuptools import setup, find_packages
from Cython.Build import cythonize 

setup(
    name='metmhn',
    packages=find_packages(),
    ext_modules = cythonize("src/perf.pyx")
)