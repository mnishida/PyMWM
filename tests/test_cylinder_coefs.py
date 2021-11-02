import unittest

import numpy as np
import numpy.testing as npt

import pymwm


class TestCylinderCoefs(unittest.TestCase):
    def setUp(self):
        self.params = {
            "core": {"shape": "cylinder", "size": 0.15, "fill": {"RI": 1.0}},
            "clad": {"book": "Au", "page": "Stewart-DLF"},
            "bounds": {"wl_max": 5.0, "wl_min": 1.0, "wl_imag": 100.0},
            "modes": {
                "wl_max": 5.0,
                "wl_min": 1.0,
                "wl_imag": 50.0,
                "num_n": 6,
                "num_m": 2,
                "ls": ["h", "v"],
            },
        }
        self.pec = {"PEC": True}

    def test_props(self):
        params = self.params.copy()
        wg = pymwm.create(params)
        wr = 2.0 * np.pi
        wi = -0.002
        w = wr + wi * 1j
        hs1, us1, vs1, jus1, jpus1, kvs1, kpvs1, As1, Bs1, Ys1 = wg.props_numpy(w)
        hs2, us2, vs2, jus2, jpus2, kvs2, kpvs2, As2, Bs2, Ys2 = wg.props(w)
        print(As1, As2)
        npt.assert_allclose(hs1, hs2)
        npt.assert_allclose(us1, us2)
        npt.assert_allclose(vs1, vs2)
        npt.assert_allclose(jus1, jus2)
        npt.assert_allclose(jpus1, jpus2)
        npt.assert_allclose(kvs1, kvs2)
        npt.assert_allclose(kpvs1, kpvs2)
        npt.assert_allclose(As1, As2)
        npt.assert_allclose(Bs1, Bs2)
        npt.assert_allclose(Ys1, Ys2)

        params["clad"] = self.pec
        wg = pymwm.create(params)
        # print(type(wg.clad))
        hs1, us1, vs1, jus1, jpus1, kvs1, kpvs1, As1, Bs1, Ys1 = wg.props_numpy(w)
        hs2, us2, vs2, jus2, jpus2, kvs2, kpvs2, As2, Bs2, Ys2 = wg.props(w)
        print(As1, As2)
        npt.assert_allclose(hs1, hs2)
        npt.assert_allclose(us1, us2)
        npt.assert_allclose(vs1, vs2)
        npt.assert_allclose(jus1, jus2)
        npt.assert_allclose(jpus1, jpus2)
        npt.assert_allclose(kvs1, kvs2)
        npt.assert_allclose(kpvs1, kpvs2)
        npt.assert_allclose(As1, As2)
        npt.assert_allclose(Bs1, Bs2)
        npt.assert_allclose(Ys1, Ys2)

    def test_norm(self):
        params = self.params.copy()
        wg = pymwm.create(params)
        wr = 2.0 * np.pi
        wi = -0.002
        w = complex(wr, wi)
        hs = np.array([wg.beta(w, alpha) for alpha in wg.alpha_all])
        As, Bs = wg.coefs(hs, w)
        for h, a, b, s, n, m in zip(hs, As, Bs, wg.s_all, wg.n_all, wg.m_all):
            pol = "E" if s == 0 else "M"
            norm = wg.norm(w, h, (pol, n, m), a, b)
            self.assertAlmostEqual(norm, 1.0)

        params["clad"] = self.pec
        wg = pymwm.create(params)
        hs = np.array([wg.beta(w, alpha) for alpha in wg.alpha_all])
        As, Bs = wg.coefs(hs, w)
        for h, a, b, s, n, m in zip(hs, As, Bs, wg.s_all, wg.n_all, wg.m_all):
            pol = "E" if s == 0 else "M"
            norm = wg.norm(w, h, (pol, n, m), a, b)
            self.assertAlmostEqual(norm, 1.0)


if __name__ == "__main__":
    unittest.main()
