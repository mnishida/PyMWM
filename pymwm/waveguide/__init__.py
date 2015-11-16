# -*- coding: utf-8 -*-


def create(params):
    if params['core']['shape'] == 'cylinder':
        from pymwm.waveguide.cylinder import Cylinder
        return Cylinder(params)
    else:
        raise ValueError("Only 'cylinder' can be chosen for 'shape'"
                         "at this time.")
