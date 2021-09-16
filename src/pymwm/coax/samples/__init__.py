# -*- coding: utf-8 -*-
import cmath
from typing import Dict, List, Tuple

import numpy as np
import scipy.optimize as so
import scipy.special as sp
from riip import Material

import pymwm
from pymwm.cutoff import Cutoff
from pymwm.waveguide import Sampling


class Samples(Sampling):
    """A class defining samples of phase constants of coaxial waveguide
    modes.

    Attributes:
        ri: A float indicating inner radius [um].
        r: A float indicating outer radius [um].
    """

    def __init__(
        self, size: float, fill: Material, clad: Material, params: Dict, size2: float
    ):
        """Init Samples class.

        Args:
            size: A float indicating the outer radius [um]
            fill: An instance of Material class for the core
            clad: An instance of Material class for the clad
            params: A dict whose keys and values are as follows:
                'wl_max': A float indicating the maximum wavelength [um]
                    (default: 5.0)
                'wl_min': A float indicating the minimum wavelength [um]
                    (default: 0.4)
                'wl_imag': A float indicating the minimum value of
                    abs(c / f_imag) [um] where f_imag is the imaginary part of
                    the frequency. (default: 5.0)
                'dw': A float indicating frequency interval
                    [rad c / 1um]=[2.99792458e14 rad / s] (default: 1 / 64).
                'num_n': An integer indicating the number of orders of modes.
                'num_m': An integer indicating the number of modes in each
                    order and polarization.
            size2: A float indicating the inner radius [um]
        """
        self.r = size
        self.ri = size2
        super().__init__(size, fill, clad, params, size2)
        self.r_ratio = self.ri / self.r
        num_n = self.params["num_n"]
        num_m = self.params["num_m"]
        co = Cutoff(num_n, num_m)
        self.co_list = []
        for n in range(num_n):
            co_per_n = []
            for pol in ["M", "E"]:
                for m in range(1, num_m + 1):
                    alpha = (pol, n, m)
                    co_per_n.append(co(alpha, self.r_ratio))
            self.co_list.append(np.array(co_per_n))

    @property
    def shape(self):
        return "coax"

    @property
    def num_all(self):
        return self.params["num_n"] * (2 * self.params["num_m"] + 1)

    def beta2_pec(self, w, n):
        """Return squares of phase constants for a PEC waveguide

        Args:
            w: A complex indicating the angular frequency
            n: A integer indicating the order of the modes.
        Returns:
            h2s: A 1D array indicating squares of phase constants, whose first
                num_m+1 elements are for TM-like modes and the rest are for
                TE-like modes.
        """
        return self.fill(w) * w ** 2 - self.co_list[n] ** 2 / self.r ** 2

    def u(self, h2: complex, w: complex, e2: complex) -> complex:
        # return cmath.sqrt(e2 * w ** 2 - h2) * self.r
        return (1 - 1j) * cmath.sqrt(0.5j * (h2 - e2 * w ** 2)) * self.ri

    def v(self, h2: complex, w: complex, e1: complex) -> complex:
        return (1 + 1j) * cmath.sqrt(-0.5j * (e1 * w ** 2 - h2)) * self.ri

    def x(self, h2: complex, w: complex, e1: complex) -> complex:
        return (1 + 1j) * cmath.sqrt(-0.5j * (e1 * w ** 2 - h2)) * self.r

    def y(self, h2: complex, w: complex, e2: complex) -> complex:
        return (1 - 1j) * cmath.sqrt(0.5j * (h2 - e2 * w ** 2)) * self.r

    def eig_eq(
        self, h2: complex, w: complex, pol: str, n: int, e1: complex, e2: complex
    ):
        """Return the value of the characteristic equation

        Args:
            h2: The square of the phase constant.
            w: The angular frequency
            pol: The polarization
            n: The order of the modes
            e1: The permittivity of the core
            e2: The permittivity of the clad.
        Returns:
            val: A complex indicating the left-hand value of the characteristic
                equation.
        """
        h2c = h2.real + 1j * h2.imag
        w2 = w ** 2
        u = self.u(h2c, w, e2)
        v = self.v(h2c, w, e1)
        x = self.x(h2c, w, e1)
        y = self.y(h2c, w, e2)

        def F(f_name, z):
            sign = 1 if f_name == "iv" else -1
            func = eval("sp." + f_name)
            f = func(n, z)
            fp = sign * func(n + 1, z) + n / z * f
            return f, fp / f

        jv, Fjv = F("jv", v)
        jx, Fjx = F("jv", x)
        yv, Fyv = F("yv", v)
        yx, Fyx = F("yv", x)
        iu, Fiu = F("iv", u)
        ky, Fky = F("kv", y)

        nuv = n * (u / v + v / u)
        nxy = n * (x / y + y / x)

        a11 = (v * Fiu + u * Fjv) / yv
        a12 = v * Fiu + u * Fyv
        a13 = nuv / yv
        a14 = nuv

        a21 = h2c * nuv / e2 / yv
        a22 = h2c * nuv / e2
        a23 = w2 * (v * Fiu + e1 / e2 * u * Fjv) / yv
        a24 = w2 * (v * Fiu + e1 / e2 * u * Fyv)

        a31 = (x * Fky + y * Fjx) / yx
        a32 = (x * Fky + y * Fyx) * jv / jx
        a33 = nxy / yx
        a34 = nxy * jv / jx

        a41 = h2c * nxy / e2 / yx
        a42 = h2c * nxy / e2 * jv / jx
        a43 = w ** 2 * (x * Fky + e1 / e2 * y * Fjx) / yx
        a44 = w ** 2 * (x * Fky + e1 / e2 * y * Fyx) * jv / jx

        return np.linalg.det(
            np.array(
                [
                    [a11, a12, a13, a14],
                    [a21, a22, a23, a24],
                    [a31, a32, a33, a34],
                    [a41, a42, a43, a44],
                ]
            )
        )

    def beta2(self, w, n, e1, e2, xis):
        """Return roots and convergences of the characteristic equation

        Args:
            w: A complex indicating the angular frequency.
            n: A integer indicating the order of the mode
            e1: A complex indicating the permittiity of tha core.
            e2: A complex indicating the permittivity of tha clad.
            xis: A complex indicating the initial approximations for the roots
                whose number of elements is 2*num_m+1.
        Returns:
            xs: A 1D array indicating the roots, whose length is 2*num_m+1.
            success: A 1D array indicating the convergence information for xs.
        """
        if self.clad.label == "PEC":
            xs = self.beta2_pec(w, n)
            success = np.ones_like(xs, dtype=bool)
            return xs, success
        from scipy.optimize import root

        num_m = self.params["num_m"]
        roots = []
        vals = []
        success = []

        def func(h2vec, *pars):
            h2 = h2vec[0] + h2vec[1] * 1j
            f = self.eig_eq(h2, *pars)
            prod_denom = 1.0
            for h2_0 in roots:
                denom = h2 - h2_0
                while abs(denom) < 1e-14:
                    denom += 1.0e-14
                prod_denom *= 1.0 / denom
            f *= prod_denom
            f_array = np.array([f.real, f.imag])
            return f_array

        for i, xi in enumerate(xis):
            if i < num_m:
                args = (w, "M", n, e1, e2)
            else:
                args = (w, "E", n, e1, e2)
            # result = root(func, (xi.real, xi.imag), args=args, jac=True,
            #               method='hybr', options={'xtol': 1.0e-9})
            result = root(
                func,
                np.array([xi.real, xi.imag]),
                args=args,
                jac=False,
                method="hybr",
                options={"xtol": 1.0e-9},
            )
            x = result.x[0] + result.x[1] * 1j
            v = self.v(x, w, e2)
            if result.success:
                roots.append(x)
            # if v.real > 0.0 and v.real > abs(v.imag):
            if v.real > 0.0:
                success.append(result.success)
            else:
                success.append(False)
            vals.append(x)
        return np.array(vals), success

    def beta2_w_min(self, n):
        """Return roots and convergences of the characteristic equation at
            the lowest angular frequency, ws[0].

        Args:
            n: A integer indicating the order of the mode
        Returns:
            xs: A 1D array indicating the roots, whose length is 2*num_m+1.
            success: A 1D array indicating the convergence information for xs.
        """
        w_0 = 0.1
        if self.clad.label == "PEC":
            xs = self.beta2_pec(self.ws[0], n)
            success = np.ones_like(xs, dtype=bool)
            return xs, success
        e1 = self.fill(w_0)
        e2_0 = -5.0e5 + self.clad(w_0).imag * 1j
        de2 = (self.clad(w_0) - e2_0) / 1000
        xis = xs = self.beta2_pec(w_0, n)
        success = np.ones_like(xs, dtype=bool)
        for i in range(1001):
            e2 = e2_0 + de2 * i
            xs, success = self.beta2(w_0, n, e1, e2, xis)
            for _, ok in enumerate(success):
                if not ok:
                    xs[_] = xis[_]
            xis = xs
        dw = (self.ws[0] - w_0) / 1000
        for i in range(1001):
            w = w_0 + dw * i
            e1 = self.fill(w)
            e2 = self.clad(w)
            xs, success = self.beta2(w, n, e1, e2, xis)
            for _, ok in enumerate(success):
                if not ok:
                    xs[_] = xis[_]
            xis = xs
        return xs, success

    @staticmethod
    def beta_from_beta2(x):
        return (1 + 1j) * np.sqrt(-0.5j * x)
        # val = np.sqrt(x)
        # if ((abs(val.real) > abs(val.imag) and val.real < 0) or
        #    (abs(val.real) < abs(val.imag) and val.imag < 0)):
        #     val *= -1
        # return val

    def __call__(self, n: int):
        """Return a dict of the roots of the characteristic equation

        Args:
            n: A integer indicating the order of the mode
        Returns:
            betas: A dict containing arrays of roots, whose key is as follows:
                (pol, n, m):
                    pol: 'E' or 'M' indicating the polarization.
                    n: A integer indicating the order of the mode.
                    m: A integer indicating the ordinal of the mode in the same
                        order.
            convs: A dict containing the convergence information for betas,
                whose key is the same as above.
        """
        num_m = self.params["num_m"]
        xs_array = np.zeros((len(self.ws), len(self.wis), 2 * num_m + 1), dtype=complex)
        success_array = np.zeros(
            (len(self.ws), len(self.wis), 2 * num_m + 1), dtype=bool
        )
        iwr = iwi = 0
        wi = self.wis[iwi]
        xis, success = self.beta2_w_min(n)
        xs_array[iwr, iwi] = xis
        success_array[iwr, iwi] = success
        for iwr in range(1, len(self.ws)):
            wr = self.ws[iwr]
            w = wr + 1j * wi
            e1 = self.fill(w)
            e2 = self.clad(w)
            xs, success = self.beta2(w, n, e1, e2, xis)
            for i, ok in enumerate(success):
                # if ok:
                #     if abs(xs[i] - xis[i]) > max(0.1 * abs(xis[i]), 5.0):
                #         success[i] = False
                #         xs[i] = xis[i]
                # else:
                if not ok:
                    xs[i] = xis[i]
            xs_array[iwr, iwi] = xs
            success_array[iwr, iwi] = success
            xis = xs
            # print(iwr, iwi, success)
        for iwi in range(1, len(self.wis)):
            wi = self.wis[iwi]
            for iwr in range(len(self.ws)):
                wr = self.ws[iwr]
                w = wr + 1j * wi
                e1 = self.fill(w)
                e2 = self.clad(w)
                if iwr == 0:
                    xis = xs_array[iwr, iwi - 1]
                else:
                    xis = (
                        xs_array[iwr, iwi - 1]
                        + xs_array[iwr - 1, iwi]
                        - xs_array[iwr - 1, iwi - 1]
                    )
                xs, success = self.beta2(w, n, e1, e2, xis)
                for i, ok in enumerate(success):
                    # if ok:
                    #     if abs(xs[i] - xis[i]) > max(
                    #             0.1 * abs(xis[i]), 5.0):
                    #         success[i] = False
                    #         xs[i] = xis[i]
                    # else:
                    if not ok:
                        xs[i] = xis[i]
                xs_array[iwr, iwi] = xs
                success_array[iwr, iwi] = success
                # print(iwr, iwi, success)
        return xs_array, success_array

    def betas_convs(self, xs_success_list):
        num_n = self.params["num_n"]
        num_m = self.params["num_m"]
        betas = {}
        convs = {}
        for n in range(num_n):
            xs_array, success_array = xs_success_list[n]
            for m in range(1, num_m + 2):
                betas[("M", n, m)] = np.zeros(
                    (len(self.ws), len(self.wis)), dtype=complex
                )
                convs[("M", n, m)] = np.zeros((len(self.ws), len(self.wis)), dtype=bool)
            for m in range(1, num_m + 1):
                betas[("E", n, m)] = np.zeros(
                    (len(self.ws), len(self.wis)), dtype=complex
                )
                convs[("E", n, m)] = np.zeros((len(self.ws), len(self.wis)), dtype=bool)
            for iwi in range(len(self.wis)):
                for iwr in range(len(self.ws)):
                    w = self.ws[iwr] + 1j * self.wis[iwi]
                    e2 = self.clad(w)
                    for i in range(num_m + 1):
                        x = xs_array[iwr, iwi][i]
                        v = self.v(x, w, e2)
                        betas[("M", n, i + 1)][iwr, iwi] = self.beta_from_beta2(x)
                        convs[("M", n, i + 1)][iwr, iwi] = (
                            success_array[iwr, iwi][i]
                            if v.real > abs(v.imag)
                            else False
                        )
                    for i in range(num_m):
                        x = xs_array[iwr, iwi][i + num_m + 1]
                        v = self.v(x, w, e2)
                        betas[("E", n, i + 1)][iwr, iwi] = self.beta_from_beta2(x)
                        convs[("E", n, i + 1)][iwr, iwi] = (
                            success_array[iwr, iwi][i + num_m + 1]
                            if v.real > abs(v.imag)
                            else False
                        )
        return betas, convs


class SamplesLowLoss(Samples):
    """A class defining samples of phase constants of cylindrical waveguide
    modes in a virtual low-loss clad waveguide by subclassing the Samples
    class.

    Attributes:
        r: A float indicating the outer radius [um]
        ri: A float indicating the inner radius [um]
        fill: An instance of Material class for the core
        clad: An instance of Material class for the clad
        params: A dict whose keys and values are as follows:
            'wl_max': A float indicating the maximum wavelength [um]
            'wl_min': A float indicating the minimum wavelength [um]
            'wl_imag': A float indicating the minimum value of
                abs(c / f_imag) [um] where f_imag is the imaginary part of
                the frequency.
            'dw': A float indicating frequency interval
                [rad * c / 1um]=[2.99792458e14 rad / s].
            'num_n': An integer indicating the number of orders of modes.
    """

    def __init__(self, size, fill, clad, params, size2):
        """Init Samples class.

        Args:
            size: A float indicating the outer radius [um]
            r: A float indicating the radius of the circular cross section [um]
            fill: An instance of Material class for the core
            clad: An instance of Material class for the clad
            params: A dict whose keys and values are as follows:
                'wl_max': A float indicating the maximum wavelength [um]
                    (default: 5.0)
                'wl_min': A float indicating the minimum wavelength [um]
                    (default: 0.4)
                'wl_imag': A float indicating the minimum value of
                    abs(c / f_imag) [um] where f_imag is the imaginary part of
                    the frequency. (default: 5.0)
                'dw': A float indicating frequency interval
                    [rad c / 1um]=[2.99792458e14 rad / s] (default: 1 / 64).
                'num_n': An integer indicating the number of orders of modes.
                'num_m': An integer indicating the number of modes in each
                    order and polarization.
            size2: A float indicating the inner radius [um]
        """
        super(SamplesLowLoss, self).__init__(size, fill, clad, params, size2)

    def __call__(self, arg: Tuple[int, int, List[np.ndarray]]):
        """Return a dict of the roots of the characteristic equation

        Args:
            arg: (iwr, iwi, xis_list)
                iwr: The ordinal of the Re(w).
                iwi: The ordinal of the Im(w).
                xis_list: The initial guess of roots whose length is 2*num_m+1
        Returns:
            xs_list: A list of num_n 1D arrays indicating the roots, whose
                length is 2*num_m+1
            success_list: A list of num_n 1D arrays indicating the convergence
                information for xs, whose length is 2*num_m+1
        """
        iwr, iwi, xis_list = arg
        im_factor = self.clad.im_factor
        self.clad.im_factor = 1.0
        wr = self.ws[iwr]
        wi = self.wis[iwi]
        w = wr + 1j * wi
        e1 = self.fill(w)
        xs_list = []
        success_list = []
        for n, x0s in enumerate(xis_list):
            xis = xs = x0s
            success = np.ones_like(xs, dtype=bool)
            for i in range(1, 16):
                self.clad.im_factor = 0.7 ** i
                if i == 15 or self.clad.im_factor < im_factor:
                    self.clad.im_factor = im_factor
                e2 = self.clad(w)
                xs, success = self.beta2(w, n, e1, e2, xis)
                for _, ok in enumerate(success):
                    if not ok:
                        xs[_] = xis[_]
                xis = xs
            xs_list.append(xs)
            success_list.append(success)
        return xs_list, success_list

    def betas_convs(self, xs_success_list):
        num_iwr = len(self.ws)
        num_iwi = len(self.wis)
        num_n = self.params["num_n"]
        num_m = self.params["num_m"]
        betas = {}
        convs = {}
        for n in range(num_n):
            for m in range(1, num_m + 1):
                betas[("M", n, m)] = np.zeros(
                    (len(self.ws), len(self.wis)), dtype=complex
                )
                convs[("M", n, m)] = np.zeros((len(self.ws), len(self.wis)), dtype=bool)
            for m in range(1, num_m + 1):
                betas[("E", n, m)] = np.zeros(
                    (len(self.ws), len(self.wis)), dtype=complex
                )
                convs[("E", n, m)] = np.zeros((len(self.ws), len(self.wis)), dtype=bool)
        for iwr in range(num_iwr):
            for iwi in range(num_iwi):
                j = iwr * num_iwi + iwi
                w = self.ws[iwr] + 1j * self.wis[iwi]
                e2 = self.clad(w)
                for n in range(num_n):
                    for i in range(num_m):
                        x = xs_success_list[j][0][n][i]
                        v = self.v(x, w, e2)
                        betas[("M", n, i + 1)][iwr, iwi] = self.beta_from_beta2(x)
                        convs[("M", n, i + 1)][iwr, iwi] = (
                            xs_success_list[j][1][n][i]
                            if v.real > abs(v.imag)
                            else False
                        )
                    for i in range(num_m):
                        x = xs_success_list[j][0][n][i + num_m + 1]
                        v = self.v(x, w, e2)
                        betas[("E", n, i + 1)][iwr, iwi] = self.beta_from_beta2(x)
                        convs[("E", n, i + 1)][iwr, iwi] = (
                            xs_success_list[j][1][n][i + num_m + 1]
                            if v.real > abs(v.imag)
                            else False
                        )
        return betas, convs
