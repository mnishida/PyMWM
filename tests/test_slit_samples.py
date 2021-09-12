#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
from multiprocessing import Pool

import numpy as np
import numpy.testing as npt
from riip import Material

from pymwm.slit.samples import Samples


class TestSlitSamples(unittest.TestCase):
    def setUp(self):
        self.betas = np.array(
            [
                [
                    1.3376125743759562 + 0.011564042849255995j,
                    0.0027442447191681966 + 10.375841268671248j,
                    0.00097855383808055987 + 20.896897655103722j,
                    0.00011573994545347484 + 31.385614129413867j,
                    -0.00087380515924567834 + 41.866423036789818j,
                    -0.0023972370044439842 + 52.343328327620277j,
                ],
                [
                    0.1,
                    0.16144417878734335 + 9.052520291531513j,
                    0.3382796371471994 + 18.186379931569725j,
                    0.5625919070749846 + 27.18042875785826j,
                    0.9111719350975946 + 35.945442096320754j,
                    1.732101248998277 + 44.2903211828822j,
                ],
            ]
        )
        self.convs = np.array(
            [
                [True, True, True, True, True, False],
                [False, True, True, True, True, True],
            ]
        )
        self.params = {
            "core": {"shape": "slit", "size": 0.3, "fill": {"RI": 1.0}},
            "clad": {"model": "gold_dl", "bound_check": False},
            "modes": {
                "wl_max": 5.0,
                "wl_min": 1.0,
                "wl_imag": 50.0,
                "dw": 1.0 / 64,
                "num_n": 6,
            },
        }

    def test_attributes(self):
        params = self.params.copy()
        r = params["core"]["size"]
        fill = Material(params["core"]["fill"])
        clad = Material(params["clad"])
        wg = Samples(r, fill, clad, params["modes"])
        p = params["modes"]
        ind_w_min = int(np.floor(2 * np.pi / p["wl_max"] / p["dw"]))
        ind_w_max = int(np.ceil(2 * np.pi / p["wl_min"] / p["dw"]))
        ind_w_imag = int(np.ceil(2 * np.pi / p["wl_imag"] / p["dw"]))
        ws = np.arange(ind_w_min, ind_w_max + 1) * p["dw"]
        wis = -np.arange(ind_w_imag + 1) * p["dw"]
        print(ws.shape, wg.ws.shape)
        npt.assert_equal(wg.ws, ws)
        npt.assert_equal(wg.wis, wis)

    @staticmethod
    def beta2_pec(w: complex, n: np.ndarray, e1: complex, r: float) -> np.ndarray:
        h2 = e1 * w ** 2 - (n * np.pi / r) ** 2
        return h2

    def test_beta2_pec(self):
        params = self.params.copy()
        r = params["core"]["size"]
        fill = Material(params["core"]["fill"])
        clad = Material(params["clad"])
        print(params["modes"])
        wg = Samples(r, fill, clad, params["modes"])
        w = 2 * np.pi / 5.0
        pec = self.beta2_pec(w, np.arange(6), fill(w), params["core"]["size"])
        npt.assert_allclose(wg.beta2_pec(w, 6), pec)

    @staticmethod
    def func(args):
        wg, pol, num_n = args
        return wg.beta2_w_min(pol, num_n)

    def test_beta2_w_min(self):
        params = self.params.copy()
        r = params["core"]["size"]
        fill = Material(params["core"]["fill"])
        clad = Material(params["clad"])
        wg = Samples(r, fill, clad, params["modes"])
        self.assertEqual(wg.ws[0], 1.25)
        num_n = params["modes"]["num_n"]
        p = Pool(2)
        args = [(wg, "M", num_n), (wg, "E", num_n)]
        vals = p.map(self.func, args)
        for i in range(2):
            h2s, success = vals[i]
            for j in range(6):
                self.assertAlmostEqual(h2s[j], self.betas[i][j] ** 2)
                # self.assertEqual(success[j], self.convs[i][j])

    def test_db(self):
        params = self.params.copy()
        r = params["core"]["size"]
        fill = Material(params["core"]["fill"])
        clad = Material(params["clad"])
        wg = Samples(r, fill, clad, params["modes"])
        try:
            betas, convs = wg.database.load()
        except IndexError:
            num_n = params["modes"]["num_n"]
            p = Pool(2)
            xs_success_list = p.map(wg, [("M", num_n), ("E", num_n)])
            betas, convs = wg.betas_convs(xs_success_list)
            wg.database.save(betas, convs)
        for n in range(6):
            npt.assert_allclose(
                [betas[("M", n, 1)][0, 0], betas[("E", n, 1)][0, 0]],
                [self.betas[0][n], self.betas[1][n]],
            )
            self.assertEqual(
                [convs[("M", n, 1)][0, 0], convs[("E", n, 1)][0, 0]],
                [self.convs[0][n], self.convs[1][n]],
            )

    def test_interpolation(self):
        params = self.params.copy()
        r = params["core"]["size"]
        fill = Material(params["core"]["fill"])
        clad = Material(params["clad"])
        wg = Samples(r, fill, clad, params["modes"])
        try:
            betas, convs = wg.database.load()
        except IndexError:
            num_n = params["modes"]["num_n"]
            p = Pool(num_n)
            xs_success_list = p.map(wg, range(num_n))
            betas, convs = wg.betas_convs(xs_success_list)
            wg.database.save(betas, convs)
        beta_funcs = wg.database.interpolation(
            betas, convs, bounds={"wl_max": 3.0, "wl_min": 1.0, "wl_imag": 50.0}
        )
        self.assertAlmostEqual(
            beta_funcs[(("M", 0, 1), "real")](2 * np.pi, 0.0)[0, 0], 6.7809313363538424
        )
        self.assertAlmostEqual(
            beta_funcs[(("M", 0, 1), "imag")](2 * np.pi, 0.0)[0, 0], 0.01839788
        )
        self.assertAlmostEqual(
            beta_funcs[(("M", 1, 1), "real")](2 * np.pi, 0.0)[0, 0], 0.02905019
        )
        self.assertAlmostEqual(
            beta_funcs[(("M", 1, 1), "imag")](2 * np.pi, 0.0)[0, 0], 7.6016237505487076
        )
        self.assertAlmostEqual(
            beta_funcs[(("M", 2, 1), "real")](2 * np.pi, 0.0)[0, 0], 0.00734511
        )
        self.assertAlmostEqual(
            beta_funcs[(("M", 2, 1), "imag")](2 * np.pi, 0.0)[0, 0], 19.703086274534229
        )
        self.assertAlmostEqual(
            beta_funcs[(("M", 3, 1), "real")](2 * np.pi, 0.0)[0, 0], -0.00016907
        )
        self.assertAlmostEqual(
            beta_funcs[(("M", 3, 1), "imag")](2 * np.pi, 0.0)[0, 0], 30.64071297
        )
        self.assertAlmostEqual(
            beta_funcs[(("E", 1, 1), "real")](2 * np.pi, 0.0)[0, 0], 0.05963503
        )
        self.assertAlmostEqual(
            beta_funcs[(("E", 1, 1), "imag")](2 * np.pi, 0.0)[0, 0], 6.5393800946369307
        )
        self.assertAlmostEqual(
            beta_funcs[(("E", 2, 1), "real")](2 * np.pi, 0.0)[0, 0], 0.09734926449677761
        )
        self.assertAlmostEqual(
            beta_funcs[(("E", 2, 1), "imag")](2 * np.pi, 0.0)[0, 0], 16.952792082725754
        )
        self.assertAlmostEqual(
            beta_funcs[(("E", 3, 1), "real")](2 * np.pi, 0.0)[0, 0], 0.15859938630326686
        )
        self.assertAlmostEqual(
            beta_funcs[(("E", 3, 1), "imag")](2 * np.pi, 0.0)[0, 0], 26.207933276759146
        )
        # self.assertAlmostEqual(
        #     beta_funcs[(('E', 4, 1), 'real')](2 * np.pi, 0.0)[0, 0],
        #     0.26532169)
        # self.assertAlmostEqual(
        #     beta_funcs[(('E', 4, 1), 'imag')](2 * np.pi, 0.0)[0, 0],
        #     34.9463874)
