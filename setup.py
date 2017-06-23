# from setuptools import setup, find_packages
from setuptools import setup
from numpy.distutils.core import Extension
from distutils.util import get_platform
from Cython.Build import cythonize
import os
from os import path
# import subprocess
import numpy as np
import pymwm

long_description = """
PyMWM is a metallic waveguide mode solver witten in Python.
"""

# blas_params = np.__config__.blas_opt_info
# blas_library_dirs = blas_params['library_dirs']
# blas_libraries = blas_params['libraries']
# lapack_params = np.__config__.lapack_opt_info
# lapack_library_dirs = lapack_params['library_dirs']
# lapack_libraries = lapack_params['libraries']
# complex_bessel = path.join('pymwm', 'cylinder', 'utils', 'complex_bessel')
# c_complex_bessel = path.join('pymwm', 'cylinder', 'utils',
#                              'c_complex_bessel.cpp')
# c_complex_bessel_h = path.join('pymwm', 'cylinder', 'utils',
#                                'c_complex_bessel.h')
cylinder_pyx = path.join('pymwm', 'cylinder', 'utils', 'cylinder_utils.pyx')
cylinder_src = [
    # path.join('pymwm', 'cylinder', 'utils', 'bessel.pyx'),
    cylinder_pyx,
    # c_complex_bessel
]
platform = get_platform()
if platform.startswith('win'):
    extra_compile_args = []
    # extra_link_args = [
    #     path.join(complex_bessel, 'src', 'amos_iso_c_fortran_wrapper.obj'),
    #     path.join(complex_bessel, 'src', 'machine.obj'),
    #     path.join(complex_bessel, 'src', 'zbesh.obj')
    # ]
    extra_link_args = []
    mkl_include = os.path.abspath(os.path.join(
        os.sep, 'Program Files (x86)', 'IntelSWTools',
        'compilers_and_libraries', 'windows', 'mkl', 'include'))
    # intel_library_dirs = [os.path.abspath(os.path.join(
    #     os.sep, 'Program Files (x86)', 'IntelSWTools',
    #     'compilers_and_libraries', 'windows', 'compiler', 'lib',
    #     'intel64'))]
    mkl_library_dirs = [os.path.abspath(os.path.join(
        os.sep, 'Program Files (x86)', 'IntelSWTools',
        'compilers_and_libraries', 'windows', 'mkl', 'lib', 'intel64'))]
    # library_dirs = intel_library_dirs + mkl_library_dirs
    library_dirs = mkl_library_dirs
    mkl_libraries = ['mkl_rt']
    # libraries = mkl_libraries + ['gfortran']
    libraries = mkl_libraries
    # cmd = "ifort {0}.f90 /c /exe:{0}.o /O3 /QxHost".format(
    #     path.join(complex_bessel, 'src', 'amos_iso_c_fortran_wrapper'))
    # subprocess.call(cmd, shell=True)
    # cmd = "ifort {0}.for /c /exe:{0}.o /O3 /QxHost".format(
    #     path.join(complex_bessel, 'src', 'machine'))
    # subprocess.call(cmd, shell=True)
    # cmd = "ifort {0}.for /c /exe:{0}.o /O3 /QxHost".format(
    #     path.join(complex_bessel, 'src', 'zbesh'))
    # subprocess.call(cmd, shell=True)
else:
    extra_compile_args = ['-fPIC', '-m64', '-fopenmp', '-march=native', '-O3',
                          '-ftree-vectorizer-verbose=2', '-Wl,--no-as-needed']
    # extra_link_args = [
    #     '-shared',
    #     path.join(complex_bessel, 'src', 'amos_iso_c_fortran_wrapper.o'),
    #     path.join(complex_bessel, 'src', 'machine.o'),
    #     path.join(complex_bessel, 'src', 'zbesh.o')
    # ]
    extra_link_args = ['-shared']
    mkl_include = os.path.abspath(os.path.join(
        os.sep, 'opt', 'intel', 'mkl', 'include'))
    mkl_library_dirs = [os.path.abspath(os.path.join(
        os.sep, 'opt', 'intel', 'mkl', 'lib', 'intel64'))]
    mkl_libraries = ['mkl_rt']
    library_dirs = mkl_library_dirs
    # libraries = mkl_libraries + ['gfortran', 'dl', 'm']
    libraries = mkl_libraries
    # cmd = "gfortran -c {0}.f90 -o {0}.o -fPIC -m64 -march=native -O3".format(
    #     path.join(complex_bessel, 'src', 'amos_iso_c_fortran_wrapper'))
    # subprocess.call(cmd, shell=True)
    # cmd = "gfortran -c {0}.for -o {0}.o -fPIC -m64 -march=native -O3".format(
    #     path.join(complex_bessel, 'src', 'machine'))
    # subprocess.call(cmd, shell=True)
    # cmd = "gfortran -c {0}.for -o {0}.o -fPIC -m64 -march=native -O3".format(
    #     path.join(complex_bessel, 'src', 'zbesh'))
    # subprocess.call(cmd, shell=True)
slit_pyx = path.join('pymwm', 'slit', 'utils', 'slit_utils.pyx')
# library_dirs = blas_library_dirs + lapack_library_dirs
# libraries = blas_libraries + lapack_libraries + ['gfortran', 'dl', 'm']
extentions = [
    Extension("pymwm.cylinder.utils.cylinder_utils",
              sources=cylinder_src,
              depends=[],
              include_dirs=[np.get_include(), ".", mkl_include],
              # depends=[c_complex_bessel_h],
              # include_dirs=[np.get_include(), ".", mkl_include,
              #               path.join(complex_bessel, 'include')],
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args,
              library_dirs=library_dirs,
              libraries=libraries,
              language="c++"),
    Extension("pymwm.slit.utils.slit_utils",
              sources=[slit_pyx],
              depends=[],
              include_dirs=[np.get_include(), ".", mkl_include],
              extra_compile_args=extra_compile_args,
              extra_link_args=['-shared'],
              library_dirs=library_dirs,
              libraries=libraries,
              language="c++")
]
ext_modules = cythonize(extentions)

setup(name='pymwm',
      version=pymwm.__version__,
      description='A metallic waveguide mode solver',
      long_description=long_description,
      # Get more strings from
      # http://www.python.org/pypi?%3Aaction=list_classifiers
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Intended Audience :: Developers",
          "Intended Audience :: Science/Research",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3",
          "Topic :: Scientific/Engineering",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Software Development :: Libraries :: Python Modules",
      ],
      keywords='metallic waveguide mode, electromagnetism',
      author=pymwm.__author__,
      author_email='mnishida@hiroshima-u.ac.jp',
      url='https://github.com/mnishida/PyMWM',
      license=pymwm.__license__,
      packages=['pymwm', 'tests', 'examples'],
      include_package_data=True,
      data_files=[
          ('examples', [path.join('examples', 'examples.ipynb')])],
      zip_safe=False,
      install_requires=[
          'numpy',
          'scipy',
          'cython',
          'pandas',
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
