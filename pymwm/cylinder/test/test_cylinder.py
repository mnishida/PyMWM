#!/usr/bin/env python
# -*- coding: utf-8 -*-
from nose.tools import assert_equal


def test_attributes():
    import numpy as np
    import numpy.testing as npt
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


def test_Yab_pec():
    import numpy as np
    import numpy.testing as npt
    import pymwm
    params = {'core': {'shape': 'cylinder', 'size': 0.15,
                       'fill': {'model': 'air'}},
              'clad': {'model': 'pec', 'im_factor': 0.0},
              'bounds': {'lmax': 3.0, 'lmin': 0.575, 'limag': 10.0},
              'modes': {'num_n': 6, 'num_m': 2}}
    wg = pymwm.create(params)
    w = 2 * np.pi / 5.0
    alpha1 = ('E', 1, 1)
    h1 = wg.beta(w, alpha1)
    a1, b1 = wg.coef(h1, w, alpha1)
    alpha2 = ('M', 1, 1)
    h2 = wg.beta(w, alpha2)
    a2, b2 = wg.coef(h2, w, alpha2)
    s1 = 0
    s2 = 1
    l1 = l2 = 0
    n1 = n2 = 1
    m1 = m2 = 1
    npt.assert_almost_equal(wg.norm(w, h1, alpha1, a1, b1), 1.0, decimal=10)
    npt.assert_almost_equal(wg.norm(w, h2, alpha2, a2, b2), 1.0, decimal=10)
    npt.assert_almost_equal(wg.Y(w, h1, alpha1, a1, b1), h1 / w, decimal=10)
    npt.assert_almost_equal(wg.Y(w, h2, alpha2, a2, b2), w / h2, decimal=10)
    npt.assert_almost_equal(
        wg.Yab(w, h1, s1, l1, n1, m1, a1, b1,
               h1, s1, l1, n1, m1, a1, b1), h1 / w, decimal=10)
    npt.assert_almost_equal(
        wg.Yab(w, h2, s2, l2, n2, m2, a2, b2,
               h2, s2, l2, n2, m2, a2, b2), w / h2, decimal=10)
    npt.assert_almost_equal(
        wg.Yab(w, h1, s1, l1, n1, m1, a1, b1,
               h2, s2, l2, n2, m2, a2, b2), 0.0, decimal=10)


def test_Yab():
    import numpy as np
    import numpy.testing as npt
    import pymwm
    params = {'core': {'shape': 'cylinder', 'size': 0.15,
                       'fill': {'model': 'air'}},
              'clad': {'model': 'gold_dl', 'im_factor': 0.0},
              'bounds': {'lmax': 3.0, 'lmin': 0.575, 'limag': 10.0},
              'modes': {'num_n': 6, 'num_m': 2}}
    wg = pymwm.create(params)
    w = 2 * np.pi / 2.0
    alpha1 = ('E', 1, 1)
    h1 = wg.beta(w, alpha1)
    a1, b1 = wg.coef(h1, w, alpha1)
    alpha2 = ('M', 1, 1)
    h2 = wg.beta(w, alpha2)
    a2, b2 = wg.coef(h2, w, alpha2)
    s1 = 0
    s2 = 1
    l1 = l2 = 0
    n1 = n2 = 1
    m1 = m2 = 1
    npt.assert_almost_equal(wg.norm(w, h1, alpha1, a1, b1), 1.0, decimal=10)
    npt.assert_almost_equal(wg.norm(w, h2, alpha2, a2, b2), 1.0, decimal=10)
    npt.assert_almost_equal(
        wg.Yab(w, h1, s1, l1, n1, m1, a1, b1,
               h1, s1, l1, n1, m1, a1, b1), wg.Y(w, h1, alpha1, a1, b1),
        decimal=10)
    npt.assert_almost_equal(
        wg.Yab(w, h2, s2, l2, n2, m2, a2, b2,
               h2, s2, l2, n2, m2, a2, b2), wg.Y(w, h2, alpha2, a2, b2),
        decimal=10)
    print(wg.Yab(w, h1, s1, l1, n1, m1, a1, b1,
                 h2, s2, l2, n2, m2, a2, b2))
    npt.assert_almost_equal(
        wg.Yab(w, h1, s1, l1, n1, m1, a1, b1,
               h2, s2, l2, n2, m2, a2, b2), 0.0, decimal=10)
