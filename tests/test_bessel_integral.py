from __future__ import annotations

import numpy as np
import numpy.testing as npt
import scipy.special as ssp
from scipy.integrate import quad, romb

import pymwm

params: dict = {
    "core": {"shape": "coax", "r": 0.15, "ri": 0.1, "fill": {"RI": 1.0}},
    "clad": {"book": "Au", "page": "Stewart-DLF", "bound_check": False},
    "bounds": {"wl_max": 5.0, "wl_min": 1.0, "wl_imag": 50.0},
    "modes": {"wl_max": 20.0, "wl_min": 1.0, "wl_imag": 50.0, "num_n": 6, "num_m": 2},
}


def test_II():
    rmin = 0.1
    rmax = 0.2

    def F(r, n, a, b):
        if a == b:
            return (
                -(r ** 2)
                / 2
                * (
                    ssp.ivp(n, a * r) ** 2
                    - (1 + n ** 2 / (a * r) ** 2) * ssp.iv(n, a * r) ** 2
                )
            )
        else:
            return -(
                b * r * ssp.iv(n, a * r) * ssp.ivp(n, b * r)
                - a * r * ssp.ivp(n, a * r) * ssp.iv(n, b * r)
            ) / (a ** 2 - b ** 2)

    for n in range(3):
        for a in [1.0, 1.0 + 0.5j, 2.0 + 0.3j]:
            for b in [1.0, 1.0 + 0.5j, 2.0 + 0.3j]:

                def rfunc(r):
                    return (r * ssp.iv(n, a * r) * ssp.iv(n, b * r)).real

                def ifunc(r):
                    return (r * ssp.iv(n, a * r) * ssp.iv(n, b * r)).imag

                I1 = F(rmax, n, a, b) - F(rmin, n, a, b)
                I2 = quad(rfunc, rmin, rmax)[0] + 1j * quad(ifunc, rmin, rmax)[0]
                print(I1, I2)
                npt.assert_almost_equal(I1, I2)


def test_JJ():
    rmin = 0.1
    rmax = 0.2

    def F(r, n, a, b):
        if a == b:
            return (
                r ** 2
                / 2
                * (
                    ssp.jvp(n, a * r) ** 2
                    + (1 - n ** 2 / (a * r) ** 2) * ssp.jv(n, a * r) ** 2
                )
            )
        else:
            return (
                b * r * ssp.jv(n, a * r) * ssp.jvp(n, b * r)
                - a * r * ssp.jvp(n, a * r) * ssp.jv(n, b * r)
            ) / (a ** 2 - b ** 2)

    for n in range(3):
        for a in [1.0, 1.0 + 0.5j, 2.0 + 0.3j]:
            for b in [1.0, 1.0 + 0.5j, 2.0 + 0.3j]:
                print(n, a, b)

                def rfunc(r):
                    return (r * ssp.jv(n, a * r) * ssp.jv(n, b * r)).real

                def ifunc(r):
                    return (r * ssp.jv(n, a * r) * ssp.jv(n, b * r)).imag

                I1 = F(rmax, n, a, b) - F(rmin, n, a, b)
                I2 = quad(rfunc, rmin, rmax)[0] + 1j * quad(ifunc, rmin, rmax)[0]
                print(I1, I2)
                npt.assert_almost_equal(I1.real, I2.real)


def test_JY():
    rmin = 0.1
    rmax = 0.2

    def F(r, n, a, b):
        if a == b:
            return (
                r ** 2
                / 2
                * (
                    ssp.jvp(n, a * r) * ssp.yvp(n, a * r)
                    + (1 - n ** 2 / (a * r) ** 2) * ssp.jv(n, a * r) * ssp.yv(n, a * r)
                )
            )
        else:
            return (
                b * r * ssp.jv(n, a * r) * ssp.yvp(n, b * r)
                - a * r * ssp.jvp(n, a * r) * ssp.yv(n, b * r)
            ) / (a ** 2 - b ** 2)

    for n in range(3):
        for a in [1.0, 1.0 + 0.5j, 2.0 + 0.3j]:
            for b in [1.0, 1.0 + 0.5j, 2.0 + 0.3j]:
                print(n, a, b)

                def rfunc(r):
                    return (r * ssp.jv(n, a * r) * ssp.yv(n, b * r)).real

                def ifunc(r):
                    return (r * ssp.jv(n, a * r) * ssp.yv(n, b * r)).imag

                I1 = F(rmax, n, a, b) - F(rmin, n, a, b)
                I2 = quad(rfunc, rmin, rmax)[0] + 1j * quad(ifunc, rmin, rmax)[0]
                print(I1, I2)
                npt.assert_almost_equal(I1.real, I2.real)


def test_IJ():
    rmin = 0.1
    rmax = 0.2

    def F(r, n, a, b):
        return -(
            b * r * ssp.iv(n, a * r) * ssp.jvp(n, b * r)
            - a * r * ssp.ivp(n, a * r) * ssp.jv(n, b * r)
        ) / (a ** 2 + b ** 2)

    for n in range(3):
        for a in [1.0, 1.0 + 0.5j, 2.0 + 0.3j]:
            for b in [1.0, 1.0 + 0.5j, 2.0 + 0.3j]:
                print(n, a, b)

                def rfunc(r):
                    return (r * ssp.iv(n, a * r) * ssp.jv(n, b * r)).real

                def ifunc(r):
                    return (r * ssp.iv(n, a * r) * ssp.jv(n, b * r)).imag

                I1 = F(rmax, n, a, b) - F(rmin, n, a, b)
                I2 = quad(rfunc, rmin, rmax)[0] + 1j * quad(ifunc, rmin, rmax)[0]
                print(I1, I2)
                npt.assert_almost_equal(I1.real, I2.real)


def test_KJ():
    r0 = 0.1

    def F(r, n, a, b):
        return (
            b * r * ssp.kv(n, a * r) * ssp.jvp(n, b * r)
            - a * r * ssp.kvp(n, a * r) * ssp.jv(n, b * r)
        ) / (a ** 2 + b ** 2)

    for n in range(3):
        for a in [1.0, 1.0 + 0.5j, 2.0 + 0.3j]:
            for b in [1.0, 1.0 + 0.5j, 2.0 + 0.3j]:
                print(n, a, b)

                def rfunc(r):
                    return (r * ssp.kv(n, a * r) * ssp.jv(n, b * r)).real

                def ifunc(r):
                    return (r * ssp.kv(n, a * r) * ssp.jv(n, b * r)).imag

                I1 = F(r0, n, a, b)
                I2 = quad(rfunc, r0, 200)[0] + 1j * quad(ifunc, r0, 200)[0]
                print(I1, I2)
                npt.assert_almost_equal(I1.real, I2.real)


def test_KK():
    r0 = 0.1

    def F(r, n, a, b):
        if a == b:
            return (
                r ** 2
                / 2
                * (
                    ssp.kvp(n, a * r) ** 2
                    - (1 + n ** 2 / (a * r) ** 2) * ssp.kv(n, a * r) ** 2
                )
            )
        else:
            return (
                b * r * ssp.kv(n, a * r) * ssp.kvp(n, b * r)
                - a * r * ssp.kvp(n, a * r) * ssp.kv(n, b * r)
            ) / (a ** 2 - b ** 2)

    for n in range(3):
        for a in [1.0, 1.0 + 0.5j, 2.0 + 0.3j]:
            for b in [1.0, 1.0 + 0.5j, 2.0 + 0.3j]:

                def rfunc(r):
                    return (r * ssp.kv(n, a * r) * ssp.kv(n, b * r)).real

                def ifunc(r):
                    return (r * ssp.kv(n, a * r) * ssp.kv(n, b * r)).imag

                I1 = F(r0, n, a, b)
                I2 = quad(rfunc, r0, 200)[0] + 1j * quad(ifunc, r0, 200)[0]
                print(I1, I2)
                npt.assert_almost_equal(I1, I2)
