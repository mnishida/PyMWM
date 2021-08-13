# from setuptools import setup, find_packages
from setuptools import setup
from numpy.distutils.core import Extension
from distutils.util import get_platform
from Cython.Build import cythonize
import os
from os import path
import numpy as np
import pymwm

long_description = """
PyMWM is a metallic waveguide mode solver witten in Python.
"""

cylinder_pyx = path.join('pymwm', 'cylinder', 'utils', 'cylinder_utils.pyx')
cylinder_src = [
    cylinder_pyx,
]
platform = get_platform()
if platform.startswith('win'):
    extra_compile_args = []
    extra_link_args = []
    mkl_include = os.path.abspath(os.path.join(
        os.sep, 'Program Files (x86)', 'IntelSWTools',
        'compilers_and_libraries', 'windows', 'mkl', 'include'))
    mkl_library_dirs = [os.path.abspath(os.path.join(
        os.sep, 'Program Files (x86)', 'IntelSWTools',
        'compilers_and_libraries', 'windows', 'mkl', 'lib', 'intel64'))]
    library_dirs = mkl_library_dirs
    mkl_libraries = ['mkl_rt']
    libraries = mkl_libraries
else:
    extra_compile_args = ['-fPIC', '-m64', '-fopenmp', '-march=native', '-O3',
                          '-ftree-vectorizer-verbose=2', '-Wl,--no-as-needed']
    extra_link_args = ['-shared']
    mkl_include = os.path.abspath(os.path.join(
        os.sep, 'opt', 'intel', 'mkl', 'include'))
    mkl_library_dirs = [os.path.abspath(os.path.join(
        os.sep, 'opt', 'intel', 'mkl', 'lib', 'intel64'))]
    mkl_libraries = ['mkl_rt']
    library_dirs = mkl_library_dirs
    libraries = mkl_libraries
slit_pyx = path.join('pymwm', 'slit', 'utils', 'slit_utils.pyx')
extentions = [
    Extension("pymwm.cylinder.utils.cylinder_utils",
              sources=cylinder_src,
              depends=[],
              include_dirs=[np.get_include(), ".", mkl_include],
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
      author="Munehiro Nishida",
      author_email='mnishida@hiroshima-u.ac.jp',
      url='https://github.com/mnishida/PyMWM',
      license="MIT",
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
          'pytest',
          'ray',
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
