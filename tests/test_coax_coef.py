from __future__ import annotations

import numpy as np
import numpy.testing as npt
from scipy.integrate import quad

import pymwm

params: dict = {
    "core": {"shape": "coax", "r": 0.15, "ri": 0.1, "fill": {"RI": 1.0}},
    "clad": {"book": "Au", "page": "Stewart-DLF", "bound_check": False},
    "bounds": {"wl_max": 5.0, "wl_min": 1.0, "wl_imag": 50.0},
    "modes": {"wl_max": 20.0, "wl_min": 1.0, "wl_imag": 50.0, "num_n": 6, "num_m": 2},
}


def test_norm():
    wg = pymwm.create(params)
    w = 2 * np.pi / 1.0
    alpha = ("E", 1, 1)
    en = 1 if alpha[1] == 0 else 2
    h = wg.beta(w, alpha)
    coef = np.random.random(8) + 1j * np.random.random(8)
    # coef = np.arange(1, 9) * 0.1 + np.arange(1, 9) * 0.01j
    I1 = wg.norm(h, w, alpha, coef)

    def rfunc(r):
        er, ep, _ = wg.e_field_r_dep(r, w, alpha, h, coef)
        return 2 * np.pi / en * r * (er ** 2 + ep ** 2).real

    def ifunc(r):
        er, ep, _ = wg.e_field_r_dep(r, w, alpha, h, coef)
        return 2 * np.pi / en * r * (er ** 2 + ep ** 2).imag

    rmin = 0.0
    rmax = 6
    I2 = np.sqrt(
        quad(rfunc, rmin, rmax, epsabs=1e-10, epsrel=1e-10)[0]
        + 1j * quad(ifunc, rmin, rmax, epsabs=1e-10, epsrel=1e-10)[0]
    )
    print(I1, I2)
    print(rfunc(rmax), ifunc(rmax))
    npt.assert_almost_equal(I1, I2)


def test_Y():
    wg = pymwm.create(params)
    w = 2 * np.pi / 1.0
    alpha = ("E", 1, 1)
    en = 1 if alpha[1] == 0 else 2
    h = wg.beta(w, alpha)
    coef = np.random.random(8) + 1j * np.random.random(8)
    Y1 = wg.Y(w, h, alpha, coef)

    def rfunc(r):
        er, ep, _, hr, hp, _ = wg.field_r_dep(r, w, alpha, h, coef)
        return 2 * np.pi / en * r * (hp * er - hr * ep).real

    def ifunc(r):
        er, ep, _, hr, hp, _ = wg.field_r_dep(r, w, alpha, h, coef)
        return 2 * np.pi / en * r * (hp * er - hr * ep).imag

    rmin = 0.0
    rmax = 6
    Y2 = (
        quad(rfunc, rmin, rmax, epsabs=1e-10, epsrel=1e-10)[0]
        + 1j * quad(ifunc, rmin, rmax, epsabs=1e-10, epsrel=1e-10)[0]
    )
    print(Y1, Y2)
    print(rfunc(rmax), ifunc(rmax))
    npt.assert_almost_equal(Y1, Y2)

    alpha = ("M", 0, 1)
    en = 1 if alpha[1] == 0 else 2
    h = wg.beta(w, alpha)
    coef = np.random.random(8) + 1j * np.random.random(8)
    Y1 = wg.Y(w, h, alpha, coef)
    Y2 = (
        quad(rfunc, rmin, rmax, epsabs=1e-10, epsrel=1e-10)[0]
        + 1j * quad(ifunc, rmin, rmax, epsabs=1e-10, epsrel=1e-10)[0]
    )
    print(Y1, Y2)
    print(rfunc(rmax), ifunc(rmax))
    npt.assert_almost_equal(Y1, Y2)


def test_props():
    wg = pymwm.create(params)
    w = 2 * np.pi / 1.0
    (
        hs1,
        xs1,
        ys1,
        us1,
        vs1,
        jxs1,
        jpxs1,
        yxs1,
        ypxs1,
        iys1,
        ipys1,
        jus1,
        jpus1,
        yus1,
        ypus1,
        kvs1,
        kpvs1,
        A1s1,
        B1s1,
        A2s1,
        B2s1,
        C2s1,
        D2s1,
        A3s1,
        B3s1,
        Ys1,
    ) = wg.props_numpy(w)
    (
        hs2,
        xs2,
        ys2,
        us2,
        vs2,
        jxs2,
        jpxs2,
        yxs2,
        ypxs2,
        iys2,
        ipys2,
        jus2,
        jpus2,
        yus2,
        ypus2,
        kvs2,
        kpvs2,
        A1s2,
        B1s2,
        A2s2,
        B2s2,
        C2s2,
        D2s2,
        A3s2,
        B3s2,
        Ys2,
    ) = wg.props(w)
    npt.assert_array_equal(hs1, hs2)
    npt.assert_array_almost_equal(xs1, xs2)
    npt.assert_array_almost_equal(ys1, ys2)
    npt.assert_array_almost_equal(us1, us2)
    npt.assert_array_almost_equal(vs1, vs2)
    npt.assert_array_almost_equal(jxs1, jxs2)
    npt.assert_array_almost_equal(jpxs1, jpxs2)
    npt.assert_array_almost_equal(yxs1, yxs2)
    npt.assert_array_almost_equal(ypxs1, ypxs2)
    npt.assert_array_almost_equal(iys1, iys2)
    npt.assert_array_almost_equal(ipys1, ipys2)
    npt.assert_array_almost_equal(jus1, jus2)
    npt.assert_array_almost_equal(jpus1, jpus2)
    npt.assert_array_almost_equal(yus1, yus2)
    npt.assert_array_almost_equal(ypus1, ypus2)
    npt.assert_array_almost_equal(kvs1, kvs2)
    npt.assert_array_almost_equal(kpvs1, kpvs2)
    npt.assert_allclose(A1s1, A1s2)
    npt.assert_allclose(B1s1, B1s2)
    npt.assert_allclose(A2s1, A2s2)
    npt.assert_allclose(B2s1, B2s2)
    npt.assert_allclose(C2s1, C2s2)
    npt.assert_allclose(D2s1, D2s2)
    npt.assert_allclose(A3s1, A3s2)
    npt.assert_allclose(B3s1, B3s2)
    print(Ys1)
    print(Ys2)
    npt.assert_allclose(Ys1, Ys2)


def test_continuity():
    delta = 1e-14
    wg = pymwm.create(params)
    w = 2 * np.pi / 1.0
    for alpha in wg.alphas["h"]:
        h = wg.beta(w, alpha)
        coef = wg.coef(h, w, alpha)
        ex1, ey1, ez1, hx1, hy1, hz1 = wg.fields(0, 0.1 - delta, w, "h", alpha, h, coef)
        ex2, ey2, ez2, hx2, hy2, hz2 = wg.fields(0, 0.1 + delta, w, "h", alpha, h, coef)
        ex3, ey3, ez3, hx3, hy3, hz3 = wg.fields(
            0, 0.15 - delta, w, "h", alpha, h, coef
        )
        ex4, ey4, ez4, hx4, hy4, hz4 = wg.fields(
            0, 0.15 + delta, w, "h", alpha, h, coef
        )
        npt.assert_array_almost_equal([ex1, ez1, hx1, hz1], [ex2, ez2, hx2, hz2])
        npt.assert_array_almost_equal([ex3, ez3, hx3, hz3], [ex4, ez4, hx4, hz4])
    for alpha in wg.alphas["v"]:
        h = wg.beta(w, alpha)
        coef = wg.coef(h, w, alpha)
        ex1, ey1, ez1, hx1, hy1, hz1 = wg.fields(0.1 - delta, 0, w, "h", alpha, h, coef)
        ex2, ey2, ez2, hx2, hy2, hz2 = wg.fields(0.1 + delta, 0, w, "h", alpha, h, coef)
        ex3, ey3, ez3, hx3, hy3, hz3 = wg.fields(
            0.15 - delta, 0, w, "h", alpha, h, coef
        )
        ex4, ey4, ez4, hx4, hy4, hz4 = wg.fields(
            0.15 + delta, 0, w, "h", alpha, h, coef
        )
        npt.assert_array_almost_equal([ey1, ez1, hy1, hz1], [ey2, ez2, hy2, hz2])
        npt.assert_array_almost_equal([ey3, ez3, hy3, hz3], [ey4, ez4, hy4, hz4])
