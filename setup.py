# from setuptools import setup, find_packages
import setuptools
from numpy.distutils.core import setup, Extension
from Cython.Build import cythonize
import os
import subprocess
import numpy as np

version = '0.1.0'

long_description = """
PyMWM is a metallic waveguide mode solver witten in Python.
"""
extra_compile_args = ['-fPIC', '-m64', '-fopenmp', '-march=native', '-O3',
                      '-ftree-vectorizer-verbose=2', '-Wl,--no-as-needed']
blas_params = np.__config__.blas_opt_info
blas_library_dirs = blas_params['library_dirs']
blas_libraries = blas_params['libraries']
# lapack_params = np.__config__.lapack_opt_info
# lapack_library_dirs = lapack_params['library_dirs']
# lapack_libraries = lapack_params['libraries']
complex_bessel = os.path.join('pymwm', 'cylinder', 'utils', 'complex_bessel')
c_complex_bessel = os.path.join('pymwm', 'cylinder', 'utils',
                                'c_complex_bessel.cpp')
c_complex_bessel_h = os.path.join('pymwm', 'cylinder', 'utils',
                                  'c_complex_bessel.h')
cylinder_pyx = os.path.join('pymwm', 'cylinder', 'utils', 'cylinder_utils.pyx')
cylinder_src = [
    # os.path.join('pymwm', 'cylinder', 'utils', 'bessel.pyx'),
    cylinder_pyx,
    c_complex_bessel
]
cmd = "gfortran -c {0}.f90 -o {0}.o -fPIC -m64 -march=native -O3".format(
    os.path.join(complex_bessel, 'src', 'amos_iso_c_fortran_wrapper'))
subprocess.call(cmd, shell=True)
cmd = "gfortran -c {0}.for -o {0}.o -fPIC -m64 -march=native -O3".format(
    os.path.join(complex_bessel, 'src', 'machine'))
subprocess.call(cmd, shell=True)
cmd = "gfortran -c {0}.for -o {0}.o -fPIC -m64 -march=native -O3".format(
    os.path.join(complex_bessel, 'src', 'zbesh'))
subprocess.call(cmd, shell=True)
extra_link_args = [
    '-shared',
    os.path.join(complex_bessel, 'src',
                 'amos_iso_c_fortran_wrapper.o'),
    os.path.join(complex_bessel, 'src',
                 'machine.o'),
    os.path.join(complex_bessel, 'src',
                 'zbesh.o')
]
slit_pyx = os.path.join('pymwm', 'slit', 'utils', 'slit_utils.pyx')
# library_dirs = blas_library_dirs + lapack_library_dirs
# libraries = blas_libraries + lapack_libraries + ['gfortran', 'dl', 'm']
library_dirs = blas_library_dirs
libraries = blas_libraries + ['gfortran', 'dl', 'm']
extentions = [
    Extension("pymwm.cylinder.utils.cylinder_utils",
              sources=cylinder_src,
              depends=[c_complex_bessel_h],
              include_dirs=[np.get_include(), ".",
                            os.path.join(complex_bessel, 'include')],
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args,
              library_dirs=library_dirs,
              libraries=libraries),
    Extension("pymwm.slit.utils.slit_utils",
              sources=[slit_pyx],
              depends=[],
              include_dirs=[np.get_include(), "."],
              extra_compile_args=extra_compile_args,
              extra_link_args=['-shared'],
              library_dirs=library_dirs,
              libraries=libraries)
]
ext_modules = cythonize(extentions)

setup(name='pymwm',
      version=version,
      description='A metallic waveguide mode solver',
      long_description=long_description,
      # Get more strings from
      # http://www.python.org/pypi?%3Aaction=list_classifiers
      classifiers=[
          "Development Status :: 4 - Beta",
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
      license="'GPLv2+'",
      packages=setuptools.find_packages(exclude=['ez_setup']),
      include_package_data=True,
      test_suite='nose.collector',
      zip_safe=False,
      install_requires=[
          'setuptools',
          'numpy',
          'scipy',
          'matplotlib',
          'pyoptmat',
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
