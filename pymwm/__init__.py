# -*- coding: utf-8 -*-


def create(params):
    if params['core']['shape'] == 'cylinder':
        from pymwm.cylinder import Cylinder
        return Cylinder(params)
    elif params['core']['shape'] == 'slit':
        from pymwm.slit import Slit
        return Slit(params)
    else:
        raise ValueError("Only 'cylinder' and 'slit' can be chosen for 'shape'"
                         "at this time.")
