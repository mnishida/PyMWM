#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import numpy.testing as npt

import pymwm

params = {
    "core": {"shape": "cylinder", "size": 0.15, "fill": {"RI": 1.0}},
    "clad": {"model": "gold_dl"},
    "bounds": {"wl_max": 5.0, "wl_min": 1.0, "wl_imag": 50.0},
    "modes": {"wl_max": 5.0, "wl_min": 1.0, "wl_imag": 50.0, "num_n": 6, "num_m": 2},
}


def test_attributes():
    wg = pymwm.create(params)
    assert wg.r == 0.15
    w = 2 * np.pi / 5.0
    assert wg.fill(w) == 1.0
    drude_lorentz = -1272.37592771801 + 351.25089220304176j
    npt.assert_almost_equal(wg.clad(w), drude_lorentz)
    #
    # def test_Yab_pec(self):
    #     params = self.params.copy()
    #     params['clad'] = {'model': 'pec'}
    #     wg = pymwm.create(params)
    #     w = 2 * np.pi / 5.0
    #     alpha1 = ('E', 1, 1)
    #     h1 = wg.beta(w, alpha1)
    #     a1, b1 = wg.coef(h1, w, alpha1)
    #     alpha2 = ('M', 1, 1)
    #     h2 = wg.beta(w, alpha2)
    #     a2, b2 = wg.coef(h2, w, alpha2)
    #     s1 = 0
    #     s2 = 1
    #     l1 = l2 = 0
    #     n1 = n2 = 1
    #     m1 = m2 = 1
    #     self.assertAlmostEqual(wg.norm(w, h1, alpha1, a1, b1), 1.0)
    #     self.assertAlmostEqual(wg.norm(w, h2, alpha2, a2, b2), 1.0)
    #     self.assertAlmostEqual(wg.Y(w, h1, alpha1, a1, b1), h1 / w)
    #     self.assertAlmostEqual(wg.Y(w, h2, alpha2, a2, b2), w / h2)
    #     self.assertAlmostEqual(
    #         wg.Yab(w, h1, s1, l1, n1, m1, a1, b1,
    #                h1, s1, l1, n1, m1, a1, b1), h1 / w)
    #     self.assertAlmostEqual(
    #         wg.Yab(w, h2, s2, l2, n2, m2, a2, b2,
    #                h2, s2, l2, n2, m2, a2, b2), w / h2)
    #     self.assertAlmostEqual(
    #         wg.Yab(w, h1, s1, l1, n1, m1, a1, b1,
    #                h2, s2, l2, n2, m2, a2, b2), 0.0)

    # def test_Yab_no_loss(self):
    #     params = self.params.copy()
    #     params['clad'] = {'model': 'gold_dl', 'im_factor': 0.0}
    #     wg = pymwm.create(params)
    #     w = 2 * np.pi / 2.0
    #     print(wg.clad(w))
    #     alpha1 = ('E', 1, 1)
    #     h1 = wg.beta(w, alpha1)
    #     a1, b1 = wg.coef(h1, w, alpha1)
    #     print(h1, a1, b1)
    #     alpha2 = ('M', 1, 1)
    #     h2 = wg.beta(w, alpha2)
    #     a2, b2 = wg.coef(h2, w, alpha2)
    #     print(h2, a2, b2)
    #     s1 = 0
    #     s2 = 1
    #     l1 = l2 = 0
    #     n1 = n2 = 1
    #     m1 = m2 = 1
    #     self.assertAlmostEqual(wg.norm(w, h1, alpha1, a1, b1), 1.0)
    #     self.assertAlmostEqual(wg.norm(w, h2, alpha2, a2, b2), 1.0)
    #     self.assertAlmostEqual(
    #         wg.Yab(w, h1, s1, l1, n1, m1, a1, b1,
    #                h1, s1, l1, n1, m1, a1, b1), wg.Y(w, h1, alpha1, a1, b1))
    #     self.assertAlmostEqual(
    #         wg.Yab(w, h2, s2, l2, n2, m2, a2, b2,
    #                h2, s2, l2, n2, m2, a2, b2), wg.Y(w, h2, alpha2, a2, b2))
    #     print(wg.Yab(w, h1, s1, l1, n1, m1, a1, b1,
    #                  h2, s2, l2, n2, m2, a2, b2))
    #     self.assertAlmostEqual(
    #         wg.Yab(
    #             w, h1, s1, l1, n1, m1, a1, b1, h2, s2, l2, n2, m2, a2, b2), 0.0)
    #
    # def test_Yab_with_loss(self):
    #     wg = pymwm.create(self.params)
    #     w = 2 * np.pi / 2.0
    #     print(wg.clad(w))
    #     alpha1 = ('E', 1, 1)
    #     h1 = wg.beta(w, alpha1)
    #     a1, b1 = wg.coef(h1, w, alpha1)
    #     alpha2 = ('M', 1, 1)
    #     h2 = wg.beta(w, alpha2)
    #     a2, b2 = wg.coef(h2, w, alpha2)
    #     s1 = 0
    #     s2 = 1
    #     l1 = l2 = 0
    #     n1 = n2 = 1
    #     m1 = m2 = 1
    #     self.assertAlmostEqual(wg.norm(w, h1, alpha1, a1, b1), 1.0)
    #     self.assertAlmostEqual(wg.norm(w, h2, alpha2, a2, b2), 1.0)
    #     print(wg.Yab(w, h1, s1, l1, n1, m1, a1, b1, h2, s2, l2, n2, m2, a2, b2))
    #     self.assertAlmostEqual(
    #         wg.Yab(w, h1, s1, l1, n1, m1, a1, b1,
    #                h2, s2, l2, n2, m2, a2, b2), 0.0)
    #
    # def test_Yaa_and_Y(self):
    #     wg = pymwm.create(self.params)
    #     w = 2 * np.pi / 0.575
    #     for i, alpha in enumerate(wg.alpha_all):
    #         h = wg.beta(w, alpha)
    #         a, b = wg.coef(h, w, alpha)
    #         s = wg.s_all[i]
    #         l = wg.l_all[i]
    #         n = wg.n_all[i]
    #         m = wg.m_all[i]
    #         Yab = wg.Yab(w, h, s, l, n, m, a, b, h, s, l, n, m, a, b)
    #         Y = wg.Y(w, h, alpha, a, b)
    #         print(Yab)
    #         print(Y)
    #         self.assertAlmostEqual(Yab, Y)
