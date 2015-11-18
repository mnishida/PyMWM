#!/usr/bin/env python
# -*- coding: utf-8 -*-
from nose.tools import assert_equal
import numpy as np
import numpy.testing as npt


def test_attributes():
    import pymwm
    params = {'core': {'shape': 'cylinder', 'size': 0.15,
                       'fill': {'model': 'air'}},
              'clad': {'model': 'gold_dl'},
              'bounds': {'lmax': 3.0, 'lmin': 0.575, 'limag': 10.0},
              'modes': {'num_n': 6, 'num_m': 2}}
    wg = pymwm.create(params)
    assert_equal(wg.r, 0.15)
    w = 2 * np.pi / 5.0
    assert_equal(wg.fill(w), 1.0)
    drude_lorentz = -1272.3749594350134+351.2507477484901j
    npt.assert_almost_equal(wg.clad(w), drude_lorentz, decimal=7)
