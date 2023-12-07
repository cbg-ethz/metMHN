import setuptools
from Cython.Build import cythonize 



setuptools.setup(
    name='MetaMHN',
    package_dir={
        'MetaMHN': 'src',
        'jx': 'src/jx',
        'np': 'src/np'
    },
    ext_modules = cythonize("src/perf.pyx")
)