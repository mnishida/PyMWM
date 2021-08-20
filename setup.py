import os

import numpy as np
from Cython.Distutils import build_ext
from setuptools import Extension, find_packages, setup

ext_modules = []
for shape in ["cylinder", "slit"]:
    pkg = f"pymwm.{shape}.utils.{shape}_utils"
    basename = os.path.join("pymwm", shape, "utils", f"{shape}_utils")
    e = Extension(
        pkg,
        sources=[basename + ".pyx"],
        depends=[basename + ".pxd"],
        include_dirs=[np.get_include(), "."],
        language="c++",
    )
    e.cython_directives = {"language_level": "3"}
    ext_modules.append(e)

setup(
    name="pymwm",
    version="0.1.3",
    url="https://github.com/mnishida/PyMWM",
    license="MIT",
    author="Munehiro Nishida",
    author_email="mnishida@hiroshima-u.ac.jp",
    description="A metallic waveguide mode solver",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    zip_safe=False,
    packages=find_packages(),
    include_package_data=True,
    setup_requires=["Cython", "numpy", "scipy"],
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
    cmdclass={"build_ext": build_ext},
)
