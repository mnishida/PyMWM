#!/usr/bin/env python
# -*- coding: utf-8 -*-
from nose.tools import assert_equal
import numpy as np
import numpy.testing as npt
from scipy.constants import c

BETAS = [
    np.array([
        1.33761261+0.01156405j,
        0.00274425+10.37584126j,
        0.00097855+20.89689765j,
        1.15739193e-04+31.38561413j,
        -8.73807985e-04+41.86642304j,
        -2.39724110e-03+52.34332833j]),
    np.array([
        0.1,
        0.16144428+9.05251986j,
        0.33827986+18.18637903j,
        0.56259232+27.18042727j,
        9.11172765e-01+35.94543976j,
        1.73210379e+00+44.29031772j])]
CONVS = [[True, True, True, True, True, True],
         [False, True, True, True, True, True]]


def func(args):
    wg, pol, num_n = args
    return wg.beta2_wmin(pol, num_n)


def test_attributes():
    import os
    from pyoptmat import Material
    from pymwm.slit.samples import Samples
    params = {'core': {'shape': 'slit', 'size': 0.15,
                       'fill': {'model': 'air'}},
              'clad': {'model': 'gold_dl'},
              'modes': {'lmax': 5.0, 'lmin': 0.4, 'limag': 5.0,
                        'dw': 1.0 / 64, 'num_n': 6}}
    r = params['core']['size']
    fill = Material(params['core']['fill'])
    clad = Material(params['clad'])
    wg = Samples(r, fill, clad, params['modes'])
    p = params['modes']
    ind_wmin = int(np.floor(2 * np.pi / p['lmax'] / p['dw']))
    ind_wmax = int(np.ceil(2 * np.pi / p['lmin'] / p['dw']))
    ind_wimag = int(np.ceil(2 * np.pi / p['limag'] / p['dw']))
    ws = np.arange(ind_wmin, ind_wmax + 1) * p['dw']
    wis = -np.arange(ind_wimag + 1) * p['dw']
    print(ws.shape, wg.ws.shape)
    npt.assert_equal(wg.ws, ws)
    npt.assert_equal(wg.wis, wis)
    assert_equal(
        wg.filename,
        os.path.join(os.path.expanduser('~'), '.pymwm',
                     "slit_size_0.15_core_air_clad_gold_dl.db"))


def beta2_pec(w, n, e1, r):
    h2 = e1 * w ** 2 - (n * np.pi / r) ** 2
    return h2


def test_beta2_pec():
    from pyoptmat import Material
    from pymwm.slit.samples import Samples
    params = {'core': {'shape': 'slit', 'size': 0.3,
                       'fill': {'model': 'air'}},
              'clad': {'model': 'gold_dl'},
              'modes': {'num_n': 6}}
    r = params['core']['size']
    fill = Material(params['core']['fill'])
    clad = Material(params['clad'])
    wg = Samples(r, fill, clad, params['modes'])
    w = 2 * np.pi / 5.0
    pec = beta2_pec(w, np.arange(6), fill(w), 0.3)
    npt.assert_almost_equal(wg.beta2_pec(w, 6), pec, decimal=8)


def test_beta2_wmin():
    from multiprocessing import Pool
    from pyoptmat import Material
    from pymwm.slit.samples import Samples
    params = {'core': {'shape': 'slit', 'size': 0.3,
                       'fill': {'model': 'air'}},
              'clad': {'model': 'gold_dl'},
              'modes': {'num_n': 6}}
    r = params['core']['size']
    fill = Material(params['core']['fill'])
    clad = Material(params['clad'])
    wg = Samples(r, fill, clad, params['modes'])
    assert_equal(wg.ws[0], 1.25)
    num_n = params['modes']['num_n']
    p = Pool(2)
    args = [(wg, 'M', num_n), (wg, 'E', num_n)]
    vals = p.map(func, args)
    for i in range(2):
        h2s, success = vals[i]
        npt.assert_almost_equal(h2s, BETAS[i] ** 2, decimal=6)
        assert_equal(success, CONVS[i])


def test_db():
    from multiprocessing import Pool
    from pyoptmat import Material
    from pymwm.slit.samples import Samples
    params = {'core': {'shape': 'slit', 'size': 0.3,
                       'fill': {'model': 'air'}},
              'clad': {'model': 'gold_dl'},
              'modes': {'num_n': 6}}
    r = params['core']['size']
    fill = Material(params['core']['fill'])
    clad = Material(params['clad'])
    wg = Samples(r, fill, clad, params['modes'])
    try:
        betas, convs = wg.load()
    except:
        num_n = params['modes']['num_n']
        p = Pool(2)
        betas_list = p.map(wg, [('M', num_n), ('E', num_n)])
        betas = {key: val for betas, convs in betas_list
                 for key, val in betas.items()}
        convs = {key: val for betas, convs in betas_list
                 for key, val in convs.items()}
        wg.save(betas, convs)
    for n in range(6):
        npt.assert_almost_equal(
            [betas[('M', n, 1)][0, 0], betas[('E', n, 1)][0, 0]],
            [BETAS[0][n], BETAS[1][n]], decimal=8)
        assert_equal(
            [convs[('M', n, 1)][0, 0], convs[('E', n, 1)][0, 0]],
            [CONVS[0][n], CONVS[1][n]])


def test_interpolation():
    from multiprocessing import Pool
    from pyoptmat import Material
    from pymwm.slit.samples import Samples
    params = {'core': {'shape': 'slit', 'size': 0.3,
                       'fill': {'model': 'air'}},
              'clad': {'model': 'gold_dl'},
              'modes': {'num_n': 6}}
    r = params['core']['size']
    fill = Material(params['core']['fill'])
    clad = Material(params['clad'])
    wg = Samples(r, fill, clad, params['modes'])
    try:
        betas, convs = wg.load()
    except:
        num_n = params['modes']['num_n']
        p = Pool(num_n)
        betas_list = p.map(wg, range(num_n))
        betas = {key: val for betas, convs in betas_list
                 for key, val in betas.items()}
        convs = {key: val for betas, convs in betas_list
                 for key, val in convs.items()}
        wg.save(betas, convs)
    beta_funcs = wg.interpolation(
        betas, convs, bounds={'lmax': 3.0, 'lmin': 0.575, 'limag': 10.0})
    npt.assert_almost_equal(
        beta_funcs[(('M', 0, 1), 'real')](2 * np.pi, 0.0)[0, 0],
        6.78093154, decimal=8)
    npt.assert_almost_equal(
        beta_funcs[(('M', 0, 1), 'imag')](2 * np.pi, 0.0)[0, 0],
        0.01839788, decimal=8)
    npt.assert_almost_equal(
        beta_funcs[(('M', 1, 1), 'real')](2 * np.pi, 0.0)[0, 0],
        0.02905019, decimal=8)
    npt.assert_almost_equal(
        beta_funcs[(('M', 1, 1), 'imag')](2 * np.pi, 0.0)[0, 0],
        7.60162343, decimal=8)
    npt.assert_almost_equal(
        beta_funcs[(('M', 2, 1), 'real')](2 * np.pi, 0.0)[0, 0],
        0.00734511, decimal=8)
    npt.assert_almost_equal(
        beta_funcs[(('M', 2, 1), 'imag')](2 * np.pi, 0.0)[0, 0],
        19.70308619, decimal=8)
    npt.assert_almost_equal(
        beta_funcs[(('M', 3, 1), 'real')](2 * np.pi, 0.0)[0, 0],
        -0.00016907, decimal=8)
    npt.assert_almost_equal(
        beta_funcs[(('M', 3, 1), 'imag')](2 * np.pi, 0.0)[0, 0],
        30.64071297, decimal=8)
    npt.assert_almost_equal(
        beta_funcs[(('E', 1, 1), 'real')](2 * np.pi, 0.0)[0, 0],
        0.05963503, decimal=8)
    npt.assert_almost_equal(
        beta_funcs[(('E', 1, 1), 'imag')](2 * np.pi, 0.0)[0, 0],
        6.53937945, decimal=8)
    npt.assert_almost_equal(
        beta_funcs[(('E', 2, 1), 'real')](2 * np.pi, 0.0)[0, 0],
        0.09734932, decimal=8)
    npt.assert_almost_equal(
        beta_funcs[(('E', 2, 1), 'imag')](2 * np.pi, 0.0)[0, 0],
        16.95279102, decimal=8)
    npt.assert_almost_equal(
        beta_funcs[(('E', 3, 1), 'real')](2 * np.pi, 0.0)[0, 0],
        0.15859949, decimal=8)
    npt.assert_almost_equal(
        beta_funcs[(('E', 3, 1), 'imag')](2 * np.pi, 0.0)[0, 0],
        26.20793155, decimal=8)
    npt.assert_almost_equal(
        beta_funcs[(('E', 4, 1), 'real')](2 * np.pi, 0.0)[0, 0],
        0.26532169, decimal=8)
    npt.assert_almost_equal(
        beta_funcs[(('E', 4, 1), 'imag')](2 * np.pi, 0.0)[0, 0],
        34.9463874, decimal=8)
