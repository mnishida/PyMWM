#!/usr/bin/env python
# -*- coding: utf-8 -*-
from nose.tools import assert_equal, assert_true
import numpy as np

result_drude = (-542.4000630622542+112.5774687080686j)
result_drude_lorentz = (-584.0804081464505+108.34074410238823j)


def test_material():
    from pymwm.material import Material
    from scipy.constants import c
    c0 = c * 1e-8
    C = 10 / c0
    drude = Material({'model': 'drude',
                      'e': 9.0685, 'wp': 2 * np.pi * 2.1556 * C,
                      'gp': 2 * np.pi * 1.836e-2 * C})
    drude_lorentz = Material({'model': 'drude_lorentz',
                              'e': 5.3983, 'wp': 13.978 * C,
                              'gp': 1.0334e-1 * C,
                              'ss': [2.5417 * 0.2679, 2.5417 * 0.7321],
                              'ws': [4.2739 * C, 5.2254 * C],
                              'gs': [2 * 4.3533e-1 * C, 2 * 6.6077e-1 * C]})
    air = Material({'model': 'air'})
    water = Material({'model': 'water'})
    SiN = Material({'model': 'SiN'})
    polymer = Material({'model': 'polymer'})
    SF03 = Material({'model': 'SF03'})
    KRS5 = Material({'model': 'KRS5'})
    ZnSe = Material({'model': 'ZnSe'})
    Ge = Material({'model': 'Ge'})
    gold_d = Material({'model': 'gold_d'})
    gold_dl = Material({'model': 'gold_dl'})
    w = 0.6 * np.pi
    assert_true(np.allclose(air(w), 1.0))
    assert_true(np.allclose(water(w), 1.333 ** 2))
    assert_true(np.allclose(SiN(w), 2.0 ** 2))
    assert_true(np.allclose(polymer(w), 2.26))
    assert_true(np.allclose(SF03(w), 1.84666 ** 2))
    assert_true(np.allclose(KRS5(w), 2.4 ** 2))
    assert_true(np.allclose(ZnSe(w), 2.4 ** 2))
    assert_true(np.allclose(Ge(w), 4.092 ** 2))
    print(drude(w))
    assert_true(np.allclose(drude(w), result_drude))
    assert_true(np.allclose(drude_lorentz(w), result_drude_lorentz))
    assert_true(np.allclose(gold_d(w), result_drude))
    assert_true(np.allclose(gold_dl(w), result_drude_lorentz))
