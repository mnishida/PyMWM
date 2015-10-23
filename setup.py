from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import os
import numpy

version = '0.1.0'

long_description = """
PyMWM is a metallic waveguide mode solver witten in Python and Cython.
"""
extra_compile_args = ['-fPIC', '-m64', '-fopenmp', '-march=native', '-O3',
                      '-ftree-vectorizer-verbose=2', '-Wl,--no-as-needed']
extra_link_args = ['-shared']
depends = []
mkl_include = '/opt/intel/mkl/include'
mkl_lib = '/opt/intel/mkl/lib/intel64'
library_dirs = [mkl_lib, '/usr/local/lib']
libraries = ['gsl', 'mkl_rt', 'pthread', 'gfortran', 'complex_bessel',
             'dl', 'm']
extentions = [
]
ext_modules = cythonize(extentions)

setup(name='pymwm',
      version=version,
      description='A metallic waveguide mode solver',
      long_description=long_description,
      # Get more strings from http://www.python.org/pypi?%3Aaction=list_classifiers
      classifiers=[
          "Development Status :: 1 - Planning",
          "Intended Audience :: Developers",
          "Intended Audience :: Science/Research",
          "Programming Language :: Python",
          "Topic :: Scientific/Engineering",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Software Development :: Libraries :: Python Modules",
      ],
      keywords='metallic waveguide mode, electromagnetism',
      author='Munehiro Nishida',
      author_email='mnishida@hiroshima-u.ac.jp',
      url='http://home.hiroshima-u.ac.jp/mnishida/',
      license="'MIT'",
      packages=find_packages(exclude=['ez_setup']),
      include_package_data=True,
      test_suite='nose.collector',
      zip_safe=False,
      install_requires=[
          'setuptools',
          # -*- Extra requirements: -*-
          # 'numpy>=1.7',
          # 'scipy>=0.12',
          # 'ipython>=1.0'
      ],
      entry_points="""
      # -*- Entry points: -*-
      """,
      ext_modules=ext_modules
      )
