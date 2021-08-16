import os
from distutils.util import get_platform

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

platform = get_platform()
if platform.startswith("win"):
    extra_compile_args = []
    extra_link_args = []
else:
    extra_compile_args = [
        "-fPIC",
        "-m64",
        "-fopenmp",
        "-march=native",
        "-O3",
        "-ftree-vectorizer-verbose=2",
        "-Wl,--no-as-needed",
    ]
    extra_link_args = ["-shared"]
extentions = [
    Extension(
        f"pymwm.{shape}.utils.{shape}_utils",
        sources=[os.path.join("pymwm", shape, "utils", f"{shape}_utils.pyx")],
        depends=[],
        include_dirs=[np.get_include(), "."],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=[],
        language="c++",
    )
    for shape in ["cylinder", "slit"]
]
ext_modules = cythonize(extentions, language_level="3")

setup(
    name="pymwm",
    version="0.1.0",
    url="https://github.com/mnishida/RII_Pandas",
    license="MIT",
    author="Munehiro Nishida",
    author_email="mnishida@hiroshima-u.ac.jp",
    description="A metallic waveguide mode solver",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    zip_safe=False,
    packages=find_packages(),
    install_requires=[line.strip() for line in open("requirements.txt").readlines()],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
    ],
    keywords="metallic waveguide mode, electromagnetism",
    ext_modules=ext_modules,
)
