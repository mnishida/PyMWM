import cmath

import numpy as np
import numpy.testing as npt

from pymwm.coax.samples import Samples
from pymwm.utils import coax_utils


def uo(h2: complex, w: complex, e2: complex, ri: float) -> complex:
    return (1 - 1j) * cmath.sqrt(0.5j * (h2 - e2 * w ** 2)) * ri


def vo(h2: complex, w: complex, e1: complex, ri: float) -> complex:
    return (1 + 1j) * cmath.sqrt(-0.5j * (e1 * w ** 2 - h2)) * ri


def xo(h2: complex, w: complex, e1: complex, r: float) -> complex:
    return (1 + 1j) * cmath.sqrt(-0.5j * (e1 * w ** 2 - h2)) * r


def yo(h2: complex, w: complex, e2: complex, r: float) -> complex:
    return (1 - 1j) * cmath.sqrt(0.5j * (h2 - e2 * w ** 2)) * r


def eig_eq_o(
    h2: complex,
    w: complex,
    pol: str,
    n: int,
    e1: complex,
    e2: complex,
    r: float,
    ri: float,
):
    w2 = w ** 2
    u = uo(h2, w, e2, ri)
    v = vo(h2, w, e1, ri)
    x = xo(h2, w, e1, r)
    y = yo(h2, w, e2, r)

    def F(f_name, z):
        sign = 1 if f_name == "iv" else -1
        func = eval("scipy.special." + f_name)
        f = func(n, z)
        fp = sign * func(n + 1, z) + n / z * f
        return f, fp / f

    jv, Fjv = F("jv", v)
    jx, Fjx = F("jv", x)
    yv, Fyv = F("yv", v)
    yx, Fyx = F("yv", x)
    iu, Fiu = F("iv", u)
    ky, Fky = F("kv", y)

    if n == 0:
        if pol == "M":
            val = (
                w2
                * (x * Fky + e1 / e2 * y * Fjx)
                / yx
                * w2
                * (v * Fiu + e1 / e2 * u * Fyv)
                - w2
                * (x * Fky + e1 / e2 * y * Fyx)
                * jv
                / jx
                * w2
                * (v * Fiu + e1 / e2 * u * Fjv)
                / yv
            )
            val /= w2 ** 2 * x * y / jx / yx / yv
        else:
            val = (x * Fky + y * Fjx) / yx * (v * Fiu + u * Fyv) - (
                v * Fiu + u * Fjv
            ) / yv * (x * Fky + y * Fyx) * jv / jx
            val /= x * y / jx / yx / yv
    else:
        nuv = n * (u / v + v / u)
        nxy = n * (x / y + y / x)

        a11 = (x * Fky + y * Fjx) / yx
        a12 = (x * Fky + y * Fyx) * jv / jx
        a13 = nxy / yx
        a14 = nxy * jv / jx

        a21 = (v * Fiu + u * Fjv) / yv
        a22 = v * Fiu + u * Fyv
        a23 = nuv / yv
        a24 = nuv

        a31 = h2 * nxy / e2 / yx
        a32 = h2 * nxy / e2 * jv / jx
        a33 = w2 * (x * Fky + e1 / e2 * y * Fjx) / yx
        a34 = w2 * (x * Fky + e1 / e2 * y * Fyx) * jv / jx

        a41 = h2 * nuv / e2 / yv
        a42 = h2 * nuv / e2
        a43 = w2 * (v * Fiu + e1 / e2 * u * Fjv) / yv
        a44 = w2 * (v * Fiu + e1 / e2 * u * Fyv)

        val = np.linalg.det(
            np.array(
                [
                    [a11, a12, a13, a14],
                    [a21, a22, a23, a24],
                    [a31, a32, a33, a34],
                    [a41, a42, a43, a44],
                ]
            )
        )
        val /= (w2 * x * y / jx / yx / yv) ** 2
    return val


def test_eiq_eq():
    size = 0.15
    size2 = 0.1
    fill = {"RI": 1.0}
    clad = {"book": "Au", "page": "Stewart-DLF", "bound_check": False}
    params = {
        "wl_max": 5.0,
        "wl_min": 1.0,
        "wl_imag": 50.0,
        "dw": 1.0 / 64,
        "num_n": 6,
        "num_m": 2,
    }
    roots = np.array([-1550 + 20j, -500 + 1.5j], dtype=complex)
    sm = Samples(size, fill, clad, params, size2)
    for pol in ("M", "E"):
        for n in range(5):
            w = sm.ws[0] - 0.1j
            h2 = 5 * w ** 2
            h2vec = np.array([h2.real, h2.imag])
            e1 = sm.fill(w)
            e2 = sm.clad(w)
            f1, fp1 = sm.eig_eq(h2vec, w, pol, n, e1, e2, roots)
            f2, fp2 = coax_utils.eig_eq_with_jac(
                h2vec, w, pol, n, e1, e2, size, size2, roots
            )
            print(pol, n)
            print(f1)
            print(f2)
            print(fp1)
            print(fp2)
            npt.assert_almost_equal(f1, f2)
            npt.assert_almost_equal(fp1, fp2)


def test_eiq_eq_for_min():
    size = 0.15
    size2 = 0.1
    fill = {"RI": 1.0}
    clad = {"book": "Au", "page": "Stewart-DLF", "bound_check": False}
    params = {
        "wl_max": 5.0,
        "wl_min": 1.0,
        "wl_imag": 50.0,
        "dw": 1.0 / 64,
        "num_n": 6,
        "num_m": 2,
    }
    roots = np.array([-1550 + 20j, -500 + 1.5j], dtype=complex)
    sm = Samples(size, fill, clad, params, size2)
    # for pol in ("M", "E"):
    #     for n in range(5):
    w = sm.ws[0] - 0.1j
    pol = "M"
    n = 1
    h2 = 5 * w ** 2
    h2vec = np.array([h2.real, h2.imag])
    e1 = sm.fill(w)
    e2 = sm.clad(w)
    f, fp = coax_utils.eig_eq_for_min_with_jac(
        h2vec, w, pol, n, e1, e2, size, size2, roots
    )
    f1, _ = coax_utils.eig_eq_for_min_with_jac(
        h2vec + np.array([1e-4, 0]), w, pol, n, e1, e2, size, size2, roots
    )
    f2, _ = coax_utils.eig_eq_for_min_with_jac(
        h2vec - np.array([1e-4, 0]), w, pol, n, e1, e2, size, size2, roots
    )
    f3, _ = coax_utils.eig_eq_for_min_with_jac(
        h2vec + np.array([0, 1e-4]), w, pol, n, e1, e2, size, size2, roots
    )
    f4, _ = coax_utils.eig_eq_for_min_with_jac(
        h2vec - np.array([0, 1e-4]), w, pol, n, e1, e2, size, size2, roots
    )
    fpr = (f1 - f2) / 2e-4
    fpi = (f3 - f4) / 2e-4
    print(f"{f}")
    print(f"{fp}")
    print(f"{fpr}")
    print(f"{fpi}")
    npt.assert_almost_equal(fp, np.array([fpr, fpi]))


def run1(cy):
    size = 0.15
    size2 = 0.1
    fill = {"RI": 1.0}
    clad = {"book": "Au", "page": "Stewart-DLF", "bound_check": False}
    params = {
        "wl_max": 5.0,
        "wl_min": 1.0,
        "wl_imag": 50.0,
        "dw": 1.0 / 64,
        "num_n": 6,
        "num_m": 2,
    }
    roots = np.array([-1550 + 20j, -500 + 1.5j], dtype=complex)
    sm = Samples(size, fill, clad, params, size2)
    w = sm.ws[0] - 0.1j
    h2 = 5 * w ** 2
    h2vec = np.array([h2.real, h2.imag])
    e1 = sm.fill(w)
    e2 = sm.clad(w)
    for pol in ("M", "E"):
        for n in range(5):
            if cy:
                [
                    coax_utils.eig_eq_cython(
                        h2vec, w, pol, n, e1, e2, size, size2, roots
                    )
                    for i in range(10000)
                ]
            else:
                [sm.eig_eq(h2vec, w, pol, n, e1, e2, roots) for i in range(10000)]
