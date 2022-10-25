# -*- coding: utf-8 -*-

__author__ = "Munehiro Nishida"
__version__ = "0.5.3"
__license__ = "MIT"


def create(params):
    if params["core"]["shape"] == "cylinder":
        from .cylinder import Cylinder

        return Cylinder(params)
    elif params["core"]["shape"] == "slit":
        from .slit import Slit

        return Slit(params)
    elif params["core"]["shape"] == "coax":
        from .coax import Coax

        return Coax(params)
    else:
        raise ValueError(
            "Only 'cylinder', 'slit' and 'coax' can be chosen for 'shape'"
            "at this time."
        )
