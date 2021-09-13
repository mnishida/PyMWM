#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

import numpy as np
import numpy.testing as npt

import pymwm


class TestSlitCoefs(unittest.TestCase):
    def setUp(self):
        self.params = {
            "core": {"shape": "slit", "size": 0.3, "fill": {"RI": 1.333}},
            "clad": {"book": "Au", "page": "Stewart-DLF"},
            "bounds": {"wl_max": 2.0, "wl_min": 1.0, "wl_imag": 50.0},
            "modes": {
                "wl_max": 2.5,
                "wl_min": 1.0,
                "wl_imag": 50.0,
                "num_n": 6,
                "num_m": 1,
                "ls": ["h", "v"],
            },
        }
        self.pec = {"e": -1e8}

    def test_coefs(self):
        params = self.params.copy()
        wg = pymwm.create(params)
        wr = 2.0 * np.pi
        wi = -0.002
        w = wr + wi * 1j
        alpha_all = wg.alpha_all
        hs = np.array([wg.beta(w, alpha) for alpha in alpha_all])
        As1, Bs1 = wg.coefs_numpy(hs, w)
        As2, Bs2 = wg.coefs(hs, w)
        print(As1, As2)
        npt.assert_allclose(As1, As2)
        npt.assert_allclose(Bs1, Bs2)

        params["clad"] = self.pec
        wg = pymwm.create(params)
        alpha_all = wg.alpha_all
        hs = np.array([wg.beta(w, alpha) for alpha in alpha_all])
        As1, Bs1 = wg.coefs_numpy(hs, w)
        As2, Bs2 = wg.coefs(hs, w)
        print(As1, As2)
        npt.assert_allclose(As1, As2)
        npt.assert_allclose(Bs1, Bs2)

    def test_ABY(self):
        params = self.params.copy()
        wg = pymwm.create(params)
        wr = 2.0 * np.pi
        wi = -0.002
        w = wr + wi * 1j
        hs = np.array([wg.beta(w, alpha) for alpha in wg.alpha_all])
        As1, Bs1 = wg.coefs_numpy(hs, w)
        Y1 = wg.Ys(w, hs, As1, Bs1)
        As2, Bs2, Y2 = wg.ABY(w, hs)
        npt.assert_allclose(As1, As2)
        npt.assert_allclose(Bs1, Bs2)
        print(wg.alpha_all)
        print(Y1, Y2)
        npt.assert_allclose(Y1, Y2)

        params["clad"] = self.pec
        wg = pymwm.create(params)
        hs = np.array([wg.beta(w, alpha) for alpha in wg.alpha_all])
        As1, Bs1 = wg.coefs_numpy(hs, w)
        Y1 = wg.Ys(w, hs, As1, Bs1)
        As2, Bs2, Y2 = wg.ABY(w, hs)
        npt.assert_allclose(As1, As2)
        npt.assert_allclose(Bs1, Bs2)
        npt.assert_allclose(Y1, Y2)

    def test_hABY(self):
        params = self.params.copy()
        wg = pymwm.create(params)
        wr = 2.0 * np.pi
        wi = -0.002
        w = wr + wi * 1j
        hs1 = np.array([wg.beta(w, alpha) for alpha in wg.alpha_all])
        As1, Bs1 = wg.coefs_numpy(hs1, w)
        Y1 = wg.Ys(w, hs1, As1, Bs1)
        hs2, As2, Bs2, Y2 = wg.hABY(w)
        npt.assert_allclose(As1, As2)
        npt.assert_allclose(Bs1, Bs2)
        npt.assert_allclose(Y1, Y2)

        params["clad"] = self.pec
        wg = pymwm.create(params)
        hs1 = np.array([wg.beta(w, alpha) for alpha in wg.alpha_all])
        As1, Bs1 = wg.coefs_numpy(hs1, w)
        Y1 = wg.Ys(w, hs1, As1, Bs1)
        hs2, As2, Bs2, Y2 = wg.hABY(w)
        npt.assert_allclose(As1, As2)
        npt.assert_allclose(Bs1, Bs2)
        npt.assert_allclose(Y1, Y2)

    def test_norm(self):
        params = self.params.copy()
        wg = pymwm.create(params)
        wr = 2.0 * np.pi
        wi = -0.002
        w = complex(wr, wi)
        hs = np.array([wg.beta(w, alpha) for alpha in wg.alpha_all])
        As, Bs = wg.coefs_numpy(hs, w)
        for h, a, b, s, n, m in zip(hs, As, Bs, wg.s_all, wg.n_all, wg.m_all):
            pol = "E" if s == 0 else "M"
            norm = wg.norm(w, h, (pol, n, m), a, b)
            self.assertAlmostEqual(norm, 1.0)

        params["clad"] = self.pec
        wg = pymwm.create(params)
        hs = np.array([wg.beta(w, alpha) for alpha in wg.alpha_all])
        As, Bs = wg.coefs_numpy(hs, w)
        for h, a, b, s, n, m in zip(hs, As, Bs, wg.s_all, wg.n_all, wg.m_all):
            pol = "E" if s == 0 else "M"
            norm = wg.norm(w, h, (pol, n, m), a, b)
            self.assertAlmostEqual(norm, 1.0)


if __name__ == "__main__":
    unittest.main()
