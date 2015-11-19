from setuptools import setup, find_packages

version = '0.1.0'

long_description = """
PyMWM is a metallic waveguide mode solver witten in Python.
"""

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
      packages=find_packages(exclude=['ez_setup']),
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
      )
