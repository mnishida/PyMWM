import io
import os
import re
import subprocess
import sys
from distutils.util import get_platform

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

import pymwm

dirname = os.path.dirname(__file__)


def read(filename):
    filename = os.path.join(dirname, filename)
    text_type = type("")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


def get_install_requires():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f.readlines() if not line.startswith("-")]


def mkl_link(platform):
    lib = os.path.join(sys.prefix, "lib")
    if platform == "win":
        mr1 = os.path.join(lib, "libmkl_rt.1.dll")
        mr = os.path.join(lib, "libmkl_rt.dll")
    else:
        mr1 = os.path.join(lib, "libmkl_rt.so.1")
        mr = os.path.join(lib, "libmkl_rt.so")
    try:
        os.symlink(mr1, mr)
    except OSError:
        print("libmkl_rt.so(dll) exists")


platform = get_platform()
if platform.startswith("win"):
    mkl_link("win")
    extra_compile_args = []
    extra_link_args = []
else:
    mkl_link("linux")
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
        sources=[os.path.join(dirname, "pymwm", shape, "utils", f"{shape}_utils.pyx")],
        depends=[],
        include_dirs=[np.get_include(), "."],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=["mkl_rt"],
        language="c++",
    )
    for shape in ["cylinder", "slit"]
]
ext_modules = cythonize(extentions)

setup(
    name="pymwm",
    version=pymwm.__version__,
    url="https://github.com/mnishida/RII_Pandas",
    license=pymwm.__license__,
    author=pymwm.__author__,
    author_email="mnishida@hiroshima-u.ac.jp",
    description="A metallic waveguide mode solver",
    long_description=read("README.md"),
    packages=find_packages(),
    install_requires=get_install_requires(),
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
