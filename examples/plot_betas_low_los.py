#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.constants import c
import pymwm
fmin = 2.5
fmax = 5.2
lmax = c * 1e-8 / fmin
lmin = c * 1e-8 / fmax
params = {'core': {'shape': 'cylinder', 'size': 0.15,
                   'fill': {'model': 'water'}},
          'clad': {'model': 'gold_dl', 'im_factor': 0.0625},
          'bounds': {'lmax': lmax, 'lmin': lmin, 'limag': 10.0},
          'modes': {'num_n': 6, 'num_m': 2}}
wg = pymwm.create(params)
w = 2 * np.pi
wg.plot_betas(fmin, fmax, comp='real')
wg.plot_betas(fmin, fmax, comp='imag')
