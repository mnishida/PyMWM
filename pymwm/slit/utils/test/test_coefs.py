from __future__ import division, print_function
import os
from nose.tools import assert_true
dirname = os.path.dirname(__file__)


def test_coefs():
    import numpy as np
    import pymwm
    params = {
        'core': {'shape': 'slit', 'size': 0.15,
                 'fill': {'model': 'water'}},
        'clad': {'model': 'gold_dl'},
        'bounds': {'lmax': 1.2, 'lmin': 0.545, 'limag': 5.0},
        'modes': {'num_n': 6, 'num_m': 2, 'ls': ['h', 'v']}}
    wg = pymwm.create(params)
    wr = 2.0 * np.pi
    wi = -0.002
    w = wr + wi * 1j
    alpha_all = wg.alpha_all
    hs = np.array([wg.beta(w, alpha) for alpha in alpha_all])
    As1, Bs1 = wg.coefs_numpy(hs, w)
    As2, Bs2 = wg.coefs(hs, w)
    print(As1, As2)
    assert_true(np.allclose(As1, As2))
    assert_true(np.allclose(Bs1, Bs2))

    params['clad'] = {'model': 'pec'}
    wg = pymwm.create(params)
    alpha_all = wg.alpha_all
    hs = np.array([wg.beta(w, alpha) for alpha in alpha_all])
    As1, Bs1 = wg.coefs_numpy(hs, w)
    As2, Bs2 = wg.coefs(hs, w)
    print(As1, As2)
    assert_true(np.allclose(As1, As2))
    assert_true(np.allclose(Bs1, Bs2))


def test_ABY():
    import numpy as np
    import pymwm
    params = {
        'core': {'shape': 'slit', 'size': 0.15,
                 'fill': {'model': 'water'}},
        'clad': {'model': 'gold_dl'},
        'bounds': {'lmax': 1.2, 'lmin': 0.545, 'limag': 5.0},
        'modes': {'num_n': 6, 'num_m': 2, 'ls': ['h', 'v']}}
    wg = pymwm.create(params)
    wr = 3.0 * np.pi
    wi = -0.002
    w = wr + wi * 1j
    hs = np.array([wg.beta(w, alpha)
                   for alpha in wg.alpha_all])
    As1, Bs1 = wg.coefs_numpy(hs, w)
    Y1 = wg.Ys(w, hs, As1, Bs1)
    As2, Bs2, Y2 = wg.ABY(w, hs)
    assert_true(np.allclose(As1, As2))
    assert_true(np.allclose(Bs1, Bs2))
    print(wg.alpha_all)
    print(Y1, Y2)
    assert_true(np.allclose(Y1, Y2))

    params['clad'] = {'model': 'pec'}
    wg = pymwm.create(params)
    hs = np.array([wg.beta(w, alpha)
                   for alpha in wg.alpha_all])
    As1, Bs1 = wg.coefs_numpy(hs, w)
    Y1 = wg.Ys(w, hs, As1, Bs1)
    As2, Bs2, Y2 = wg.ABY(w, hs)
    assert_true(np.allclose(As1, As2))
    assert_true(np.allclose(Bs1, Bs2))
    assert_true(np.allclose(Y1, Y2))


def test_hABY():
    import numpy as np
    import pymwm
    params = {
        'core': {'shape': 'slit', 'size': 0.15,
                 'fill': {'model': 'water'}},
        'clad': {'model': 'gold_dl'},
        'bounds': {'lmax': 1.2, 'lmin': 0.545, 'limag': 5.0},
        'modes': {'num_n': 6, 'num_m': 2, 'ls': ['h', 'v']}}
    wg = pymwm.create(params)
    wr = 3.0 * np.pi
    wi = -0.002
    w = wr + wi * 1j
    hs1 = np.array([wg.beta(w, alpha)
                   for alpha in wg.alpha_all])
    As1, Bs1 = wg.coefs_numpy(hs1, w)
    Y1 = wg.Ys(w, hs1, As1, Bs1)
    hs2, As2, Bs2, Y2 = wg.hABY(w)
    assert_true(np.allclose(As1, As2))
    assert_true(np.allclose(Bs1, Bs2))
    assert_true(np.allclose(Y1, Y2))

    params['clad'] = {'model': 'pec'}
    wg = pymwm.create(params)
    hs1 = np.array([wg.beta(w, alpha)
                   for alpha in wg.alpha_all])
    As1, Bs1 = wg.coefs_numpy(hs1, w)
    Y1 = wg.Ys(w, hs1, As1, Bs1)
    hs2, As2, Bs2, Y2 = wg.hABY(w)
    assert_true(np.allclose(As1, As2))
    assert_true(np.allclose(Bs1, Bs2))
    assert_true(np.allclose(Y1, Y2))


def test_norm():
    import numpy as np
    import pymwm
    params = {
        'core': {'shape': 'slit', 'size': 0.15,
                 'fill': {'model': 'water'}},
        'clad': {'model': 'gold_dl'},
        'bounds': {'lmax': 1.2, 'lmin': 0.545, 'limag': 5.0},
        'modes': {'num_n': 6, 'num_m': 2, 'ls': ['h', 'v']}}
    wg = pymwm.create(params)
    wr = 3.0 * np.pi
    wi = -0.002
    w = complex(wr, wi)
    hs = np.array([wg.beta(w, alpha)
                   for alpha in wg.alpha_all])
    As, Bs = wg.coefs_numpy(hs, w)
    for h, a, b, s, n, m in zip(hs, As, Bs, wg.s_all, wg.n_all, wg.m_all):
        pol = 'E' if s == 0 else 'M'
        norm = wg.norm(w, h, (pol, n, m), a, b)
        assert_true(np.allclose(norm, 1.0))

    params['clad'] = {'model': 'pec'}
    wg = pymwm.create(params)
    hs = np.array([wg.beta(w, alpha)
                   for alpha in wg.alpha_all])
    As, Bs = wg.coefs_numpy(hs, w)
    for h, a, b, s, n, m in zip(hs, As, Bs, wg.s_all, wg.n_all, wg.m_all):
        pol = 'E' if s == 0 else 'M'
        norm = wg.norm(w, h, (pol, n, m), a, b)
        assert_true(np.allclose(norm, 1.0))
