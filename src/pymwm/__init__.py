# -*- coding: utf-8 -*-

__author__ = "Munehiro Nishida"
__version__ = "0.2.8"
__license__ = "MIT"


def create(params):
    if params["core"]["shape"] == "cylinder":
        from .cylinder import Cylinder

        return Cylinder(params)
    elif params["core"]["shape"] == "slit":
        from .slit import Slit

        return Slit(params)
    else:
        raise ValueError(
            "Only 'cylinder' and 'slit' can be chosen for 'shape'" "at this time."
        )
