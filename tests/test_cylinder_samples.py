#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

import numpy as np
import numpy.testing as npt
import ray

import pymwm
from pymwm.cylinder.samples import Samples, SamplesForRay


class TestCylinderSamples(unittest.TestCase):
    def setUp(self):
        betas = []
        convs = []
        betas.append(
            [
                1.32930242e-03 + 15.97093724j,
                -4.12022699e-04 + 36.77569389j,
                -2.96423554e-03 + 57.67676409j,
                4.25706586e-01 + 22.16438948j,
                1.16961605e00 + 39.91871066j,
            ]
        )
        convs.append([True, True, True, True, True])
        betas.append(
            [
                3.93618487e-04 + 25.50725697j,
                -1.69749464e-03 + 46.75220214j,
                -3.60166883e-03 + 67.81044568j,
                1.88412925e-01 + 10.63429198j,
                6.65409650e-01 + 30.69581722j,
            ]
        )
        convs.append([True, False, True, True, True])
        betas.append(
            [
                -3.14039183e-04 + 34.21067616j,
                -3.02497952e-03 + 56.09987523j,
                -3.99382568e-03 + 77.45436569j,
                3.16581667e-01 + 17.70646046j,
                9.90337935e-01 + 38.34855698j,
            ]
        )
        convs.append([True, False, True, True, True])
        betas.append(
            [
                -1.22828011e-03 + 42.51416161j,
                -3.77291544e-03 + 65.06037821j,
                -4.27041215e-03 + 86.7578285j,
                4.45859022e-01 + 24.35935701j,
                1.56012941e00 + 45.43872731j,
            ]
        )
        convs.append([True, False, True, True, False])
        betas.append(
            [
                -0.00274348 + 50.57304098j,
                -0.00424744 + 73.75302452j,
                -0.00448273 + 95.80756518j,
                0.58332927 + 30.80613956j,
                2.57935560 + 52.37067052j,
            ]
        )
        convs.append([False, False, True, True, False])
        betas.append(
            [
                -0.00422390 + 58.46301045j,
                -0.00458645 + 82.24672285j,
                -0.00465523 + 104.65914944j,
                0.73689393 + 37.1144517j,
                3.79669182 + 59.48095715j,
            ]
        )
        convs.append([False, False, True, True, False])
        self.betas = np.array(betas)
        self.convs = np.array(convs)
        self.params = {
            "core": {"shape": "cylinder", "size": 0.15, "fill": {"RI": 1.0}},
            "clad": {"book": "Au", "page": "Stewart-DLF", "bound_check": False},
            "modes": {
                "wl_max": 5.0,
                "wl_min": 1.0,
                "wl_imag": 50.0,
                "dw": 1.0 / 64,
                "num_n": 6,
                "num_m": 2,
            },
        }

    def test_attributes(self):
        params: dict = self.params.copy()
        size: float = params["core"]["size"]
        fill = params["core"]["fill"]
        clad = params["clad"]
        wg = Samples(size, fill, clad, params["modes"])
        p = params["modes"]
        ind_w_min = int(np.floor(2 * np.pi / p["wl_max"] / p["dw"]))
        ind_w_max = int(np.ceil(2 * np.pi / p["wl_min"] / p["dw"]))
        ind_w_imag = int(np.ceil(2 * np.pi / p["wl_imag"] / p["dw"]))
        ws = np.arange(ind_w_min, ind_w_max + 1) * p["dw"]
        wis = -np.arange(ind_w_imag + 1) * p["dw"]
        npt.assert_equal(wg.ws, ws)
        npt.assert_equal(wg.wis, wis)

    def test_beta2_pec(self):
        params: dict = self.params.copy()
        size: float = params["core"]["size"]
        fill = params["core"]["fill"]
        clad = params["clad"]
        wg = Samples(size, fill, clad, params["modes"])
        w = 2 * np.pi / 5.0
        pec0 = (
            np.array(
                [
                    7.9914227540830467j,
                    18.389529559512987j,
                    28.838915878616223j,
                    12.756889235180587j,
                    23.376846509563137j,
                ]
            )
            * 2
        )
        npt.assert_almost_equal(wg.beta2_pec(w, 0), pec0 ** 2)
        pec1 = (
            np.array(
                [
                    12.756889235180589j,
                    23.376846509563137j,
                    33.90573916009118j,
                    6.1050317506026381j,
                    17.760365196297929j,
                ]
            )
            * 2
        )
        npt.assert_almost_equal(wg.beta2_pec(w, 1), pec1 ** 2)
        pec2 = (
            np.array(
                [
                    17.107206360452462j,
                    28.050444310850633j,
                    38.72770732091869j,
                    10.161382581946896j,
                    22.344945200686148j,
                ]
            )
            * 2
        )
        npt.assert_almost_equal(wg.beta2_pec(w, 2), pec2 ** 2)
        pec3 = (
            np.array(
                [
                    21.257922769420034j,
                    32.530676457118169j,
                    43.37945228513604j,
                    13.989860591754667j,
                    26.710066174072818j,
                ]
            )
            * 2
        )
        npt.assert_almost_equal(wg.beta2_pec(w, 3), pec3 ** 2)
        pec4 = (
            np.array(
                [
                    25.286669814449052j,
                    36.877012636462631j,
                    47.90433520177347j,
                    17.71403733166369j,
                    30.934940730583346j,
                ]
            )
            * 2
        )
        npt.assert_almost_equal(wg.beta2_pec(w, 4), pec4 ** 2)
        pec5 = (
            np.array(
                [
                    29.231527454256089j,
                    41.123881000095977j,
                    52.330141681593268j,
                    21.376155694373626j,
                    35.060573334299526j,
                ]
            )
            * 2
        )
        npt.assert_almost_equal(wg.beta2_pec(w, 5), pec5 ** 2)

    def test_beta2_w_min(self):
        params: dict = {
            "core": {"shape": "cylinder", "size": 0.15, "fill": {"RI": 1.0}},
            "clad": {"book": "Au", "page": "Stewart-DLF", "bound_check": False},
            "modes": {"num_n": 6, "num_m": 2},
        }
        size = params["core"]["size"]
        fill = params["core"]["fill"]
        clad = params["clad"]
        wg = Samples(size, fill, clad, params["modes"])
        self.assertEqual(wg.ws[0], 1.25)
        num_n = params["modes"]["num_n"]
        ray.shutdown()
        try:
            ray.init()
            p_modes_id = ray.put(params["modes"])
            pool = ray.util.ActorPool(
                SamplesForRay.remote(size, fill, clad, p_modes_id) for _ in range(num_n)
            )
            vals = list(
                pool.map(lambda a, arg: a.beta2_w_min.remote(arg), range(num_n))
            )
        finally:
            ray.shutdown()
        for n in range(6):
            print(n)
            h2s, success = vals[n]
            print(success)
            npt.assert_allclose(h2s, self.betas[n] * self.betas[n], rtol=1e-6)

    def test_db(self):
        params: dict = self.params.copy()
        size = params["core"]["size"]
        fill = params["core"]["fill"]
        clad = params["clad"]
        wg = Samples(size, fill, clad, params["modes"])
        try:
            betas, convs = wg.database.load()
        except IndexError:
            num_n = params["modes"]["num_n"]
            ray.shutdown()
            try:
                ray.init()
                p_modes_id = ray.put(params["modes"])
                pool = ray.util.ActorPool(
                    SamplesForRay.remote(size, fill, clad, p_modes_id)
                    for _ in range(num_n)
                )
                xs_success_list = list(
                    pool.map(lambda a, arg: a.task.remote(arg), range(num_n))
                )
            finally:
                ray.shutdown()
            betas, convs = wg.betas_convs(xs_success_list)
            wg.database.save(betas, convs)
        for n in range(6):
            print(n)
            npt.assert_allclose(
                [
                    betas[("M", n, 1)][0, 0],
                    betas[("M", n, 2)][0, 0],
                    betas[("E", n, 1)][0, 0],
                    betas[("E", n, 2)][0, 0],
                ],
                [
                    self.betas[n][0],
                    self.betas[n][1],
                    self.betas[n][3],
                    self.betas[n][4],
                ],
            )
            self.assertEqual(
                [
                    convs[("M", n, 1)][0, 0],
                    convs[("M", n, 2)][0, 0],
                    convs[("E", n, 1)][0, 0],
                    convs[("E", n, 2)][0, 0],
                ],
                [
                    self.convs[n][0],
                    self.convs[n][1],
                    self.convs[n][3],
                    self.convs[n][4],
                ],
            )

    def test_interpolation(self):
        params: dict = self.params.copy()
        size = params["core"]["size"]
        fill = params["core"]["fill"]
        clad = params["clad"]
        wg = Samples(size, fill, clad, params["modes"])
        try:
            betas, convs = wg.database.load()
        except IndexError:
            num_n = params["modes"]["num_n"]
            ray.shutdown()
            try:
                ray.init()
                p_modes_id = ray.put(params["modes"])
                pool = ray.util.ActorPool(
                    SamplesForRay.remote(size, fill, clad, p_modes_id)
                    for _ in range(num_n)
                )
                xs_success_list = list(
                    pool.map(lambda a, arg: a.task.remote(arg), range(num_n))
                )
            finally:
                ray.shutdown()
            betas, convs = wg.betas_convs(xs_success_list)
            wg.database.save(betas, convs)
        beta_funcs = wg.database.interpolation(
            betas, convs, bounds={"wl_max": 3.0, "wl_min": 1.0, "wl_imag": 100.0}
        )
        self.assertAlmostEqual(
            beta_funcs[(("M", 0, 1), "real")](2 * np.pi, 0.0)[0, 0],
            0.011030829731291485,
        )
        self.assertAlmostEqual(
            beta_funcs[(("M", 0, 1), "imag")](2 * np.pi, 0.0)[0, 0], 14.374412149329419
        )
        self.assertAlmostEqual(
            beta_funcs[(("M", 1, 1), "real")](2 * np.pi, 0.0)[0, 0],
            0.0024442176360349316,
        )
        self.assertAlmostEqual(
            beta_funcs[(("M", 1, 1), "imag")](2 * np.pi, 0.0)[0, 0], 24.573875828863905
        )
        self.assertAlmostEqual(
            beta_funcs[(("E", 1, 1), "real")](2 * np.pi, 0.0)[0, 0], 0.07516250767481286
        )
        self.assertAlmostEqual(
            beta_funcs[(("E", 1, 1), "imag")](2 * np.pi, 0.0)[0, 0], 8.2795729054555345
        )
        self.assertAlmostEqual(
            beta_funcs[(("E", 1, 2), "real")](2 * np.pi, 0.0)[0, 0], 0.1884476108781034
        )
        self.assertAlmostEqual(
            beta_funcs[(("E", 1, 2), "imag")](2 * np.pi, 0.0)[0, 0], 29.74494425189081
        )
        self.assertAlmostEqual(
            beta_funcs[(("E", 2, 1), "real")](2 * np.pi, 0.0)[0, 0], 0.10224849620607172
        )
        self.assertAlmostEqual(
            beta_funcs[(("E", 2, 1), "imag")](2 * np.pi, 0.0)[0, 0], 16.184787946722981
        )
        self.assertAlmostEqual(
            beta_funcs[(("E", 3, 1), "real")](2 * np.pi, 0.0)[0, 0], 0.1353140706821849
        )
        self.assertAlmostEqual(
            beta_funcs[(("E", 3, 1), "imag")](2 * np.pi, 0.0)[0, 0], 23.102611360449739
        )
        self.assertAlmostEqual(
            beta_funcs[(("E", 4, 1), "real")](2 * np.pi, 0.0)[0, 0], 0.1720347941605904
        )
        self.assertAlmostEqual(
            beta_funcs[(("E", 4, 1), "imag")](2 * np.pi, 0.0)[0, 0], 29.661836787913028
        )

    def test_eig_mat(self):
        size = self.params["core"]["size"]
        fill = self.params["core"]["fill"]
        clad = self.params["clad"]
        wg = Samples(size, fill, clad, self.params["modes"])
        h2 = -35513.94604091 - 6.53379717e5j
        w = 2 * np.pi / 100.0
        e1 = wg.fill(w)
        e2 = wg.clad(w) * 100
        eps = 1e-4
        a, b = wg.eig_mat(h2, w, "M", 0, e1, e2)
        a1, b1 = wg.eig_mat(h2 + eps, w, "M", 0, e1, e2)
        a2, b2 = wg.eig_mat(h2 - eps, w, "M", 0, e1, e2)
        da_dh2 = (a1 - a2) / (2 * eps)
        npt.assert_almost_equal(b, da_dh2)
        a, b = wg.eig_mat(h2, w, "E", 0, e1, e2)
        a1, b1 = wg.eig_mat(h2 + eps, w, "E", 0, e1, e2)
        a2, b2 = wg.eig_mat(h2 - eps, w, "E", 0, e1, e2)
        da_dh2 = (a1 - a2) / (2 * eps)
        npt.assert_almost_equal(b, da_dh2)
        a, b = wg.eig_mat(h2, w, "M", 2, e1, e2)
        a1, b1 = wg.eig_mat(h2 + eps, w, "M", 2, e1, e2)
        a2, b2 = wg.eig_mat(h2 - eps, w, "M", 2, e1, e2)
        da_dh2 = (a1 - a2) / (2 * eps)
        npt.assert_almost_equal(b, da_dh2)


# def test_beta2_w_min():
#     params: dict = {
#         "core": {"shape": "cylinder", "size": 0.15, "fill": {"RI": 1.0}},
#         "clad": {"book": "Au", "page": "Stewart-DLF", "bound_check": False},
#         "modes": {
#             "wl_max": 5.0,
#             "wl_min": 1.0,
#             "wl_imag": 50.0,
#             "dw": 1.0 / 64,
#             "num_n": 6,
#             "num_m": 2,
#         },
#     }
#     size: float = params["core"]["size"]
#     fill = params["core"]["fill"]
#     clad = params["clad"]
#     p = params["modes"]
#     wg = pymwm.cylinder.samples.Samples(size, fill, clad, p)

#     for n in range(6):
#         try:
#             val = wg.beta2_w_min(n)
#             # npt.assert_allclose(val, 1.0, rtol=0.05)
#             print(f"{n=}")
#             print(f"{val=}")
#         except Exception as e:
#             raise e
#     raise Exception


if __name__ == "__main__":
    unittest.main()
