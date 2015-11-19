#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pymwm
params = {'core': {'shape': 'cylinder', 'size': 0.2,
                   'fill': {'model': 'air'}},
          'clad': {'model': 'gold_dl'},
          'bounds': {'lmax': 1.2, 'lmin': 0.545, 'limag': 10.0},
          'modes': {'num_n': 6, 'num_m': 2}}
wg = pymwm.create(params)
w = 2 * np.pi
alpha = ('M', 0, 1)
wg.plot_efield(w, 'h', alpha, 0.3, 0.3)
wg.plot_efield_on_x_axis(w, 'h', alpha, 'x', 0.3)
wg.plot_hfield(w, 'h', alpha, 0.3, 0.3)
wg.plot_hfield_on_x_axis(w, 'h', alpha, 'y', 0.3)
alpha = ('H', 1, 1)
wg.plot_efield(w, 'v', alpha, 0.3, 0.3)
wg.plot_efield_on_x_axis(w, 'v', alpha, 'y', 0.3)
wg.plot_hfield(w, 'v', alpha, 0.3, 0.3)
wg.plot_hfield_on_x_axis(w, 'v', alpha, 'x', 0.3)
