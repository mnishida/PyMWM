#!/usr/bin/env python
# -*- coding: utf-8 -*-
# from nose.tools import assert_equal, assert_true
import numpy as np
import numpy.testing as npt
from pymwm.muller import Muller


def test_func1():
    vals = np.sort(
        np.around(
            [1 + 0j, (-1 + 1j * np.sqrt(3)) / 2,  (-1 - 1j * np.sqrt(3)) / 2],
            decimals=8))
    solver = Muller(lambda z: z ** 3 - 1, num_roots=len(vals), xtol=1.0e-8,
                    rtol=1.0e-5, ftol=1.0e-8, maxiter=100)
    solver()
    roots = np.sort(np.around(solver.roots, decimals=8))
    print(roots, vals)
    npt.assert_almost_equal(roots, vals, decimal=8)


def test_func2():
    vals = np.sort([0.038060233744356624, 0.30865828381745514,
                    0.6913417161825454, 0.9619397662556431])
    solver = Muller(
        lambda z: 128 * z ** 4 - 256 * z ** 3 + 160 * z ** 2 - 32 * z + 1,
        num_roots=len(vals), xtol=1.0e-8, rtol=1.0e-5, ftol=1.0e-8,
        maxiter=100)
    solver(0.0)
    roots = np.sort(np.real(solver.roots))
    print(roots, vals)
    npt.assert_almost_equal(roots, vals, decimal=8)


def test_func3():
    vals = np.sort(
        np.around(
            [0.9946779090453202 - 1.024878957947296j,
             0.9946779090453202 + 1.024878957947296j,
             3.00532209095468 - 3.9963693691499347j,
             3.00532209095468 + 3.9963693691499347j], decimals=8))
    solver = Muller(
        lambda z: z ** 4 - 8 * z ** 3 + 39 * z ** 2 - 62 * z + 51,
        num_roots=len(vals), xtol=1.0e-8, rtol=1.0e-5, ftol=1.0e-8,
        maxiter=100)
    solver()
    roots = np.sort(np.around(solver.roots, decimals=8))
    print(roots, vals)
    npt.assert_almost_equal(roots, vals, decimal=8)


def test_func4():
    vals = np.sort([-2.0, 1.0, 1.0])
    solver = Muller(
        lambda z: z ** 3 - 3 * z + 2,
        num_roots=len(vals), xtol=1.0e-8, rtol=1.0e-5, ftol=1.0e-8,
        maxiter=100)
    solver()
    roots = np.sort(np.real(solver.roots))
    print(roots, vals)
    npt.assert_almost_equal(roots, vals, decimal=8)


def test_func5():
    vals = np.sort(
        np.around(
            [-1, -0.9510565162951535 - 0.30901699437494745j,
             -0.9510565162951535 + 0.30901699437494745j,
             -0.8090169943749475 - 0.5877852522924731j,
             -0.8090169943749475 + 0.5877852522924731j,
             -0.5877852522924731 - 0.8090169943749475j,
             -0.5877852522924731 + 0.8090169943749475j,
             -0.3090169943749474 - 0.9510565162951535j,
             -0.3090169943749474 + 0.9510565162951535j,
             - 1j, 1j,
             0.30901699437494745 - 0.9510565162951535j,
             0.30901699437494745 + 0.9510565162951535j,
             0.5877852522924731 - 0.8090169943749475j,
             0.5877852522924731 + 0.8090169943749475j,
             0.8090169943749475 - 0.5877852522924731j,
             0.8090169943749475 + 0.5877852522924731j,
             0.9510565162951535 - 0.30901699437494745j,
             0.9510565162951535 + 0.30901699437494745j,
             1], decimals=8))
    solver = Muller(
        lambda z: z ** 20 - 1,
        num_roots=len(vals), xtol=1.0e-8, rtol=1.0e-5, ftol=1.0e-8,
        maxiter=100)
    solver()
    roots = np.sort(np.around(solver.roots, decimals=8))
    print(roots, vals)
    npt.assert_almost_equal(roots, vals, decimal=8)
