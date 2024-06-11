from setuptools import setup, find_packages, Extension

setup(
    name='metmhn',
    packages=find_packages(),
    ext_modules = [Extension("metmhn.int_order_conversion", ["metmhn/int_order_conversion.c"])]
)
