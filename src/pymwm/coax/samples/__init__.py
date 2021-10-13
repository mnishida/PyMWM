from __future__ import annotations

import numpy as np
import ray
import riip
import scipy.special as ssp
from scipy.optimize import minimize, root

from pymwm.cutoff import Cutoff
from pymwm.utils import coax_utils, eig_mat_utils
from pymwm.waveguide import Sampling


class Samples(Sampling):
    """A class defining samples of phase constants of coaxial waveguide
    modes.

    Attributes:
        fill: An instance of Material class for the core
        clad: An instance of Material class for the clad
        size: A float indicating the size of core [um].
        size2: A float indicating the optional size of core [um].
        params: A dict whose keys and values are as follows:
            'wl_max': A float indicating the maximum wavelength [um]
            'wl_min': A float indicating the minimum wavelength [um]
            'wl_imag': A float indicating the minimum value of
                abs(c / f_imag) [um] where f_imag is the imaginary part of
                the frequency.
            'dw': A float indicating frequency interval
                [rad * c / 1um]=[2.99792458e14 rad / s].
            'num_n': An integer indicating the number of orders of modes.
            'num_m': An integer indicating the number of modes in each
                order and polarization.
        ws: A 1D array indicating the real part of the angular frequencies
            to be calculated [rad (c / 1um)]=[2.99792458e14 rad / s].
        wis: A 1D array indicating the imaginary part of the angular
            frequencies to be calculated [rad * (c / 1um)].
        ri: A float indicating inner radius [um].
        r: A float indicating outer radius [um].
        co_list: A list indicating cutoffs for PEC waveguide.
    """

    def __init__(
        self, size: float, fill: dict, clad: dict, params: dict, size2: float
    ) -> None:
        """Init Samples class.

        Args:
            size: A float indicating the outer radius [um]
            fill (dict): Parameters for riip.Material class for the core
            clad (dict): Parameters for riip.Material class for the clad
            params (dict): Keys and values are as follows:
                'wl_max' (float): The maximum wavelength [um].
                    Defaults to 5.0.
                'wl_min' (float): The minimum wavelength [um].
                    Defaults to 0.4.
                'wl_imag' (float): The minimum value of abs(c / f_imag) [um]
                    where f_imag is the imaginary part of the frequency.
                    Defaults to 5.0.
                'dw' (float): The frequency interval [rad c / 1um]=[2.99792458e14 rad / s].
                    Defaults to 1 / 64.
                'num_n' (int): The number of orders of modes.
                'num_m' (int): The number of modes in each order and polarization.
                    (num_m + 1 for TM-like mode, num_m for TE-like mode)
            size2: A float indicating the inner radius [um]
        """
        super().__init__(size, fill, clad, params, size2)
        self.r = size
        self.ri = size2
        self.r_ratio = self.ri / self.r
        num_n = self.params["num_n"]
        num_m = self.params["num_m"]
        co = Cutoff(num_n, num_m)
        self.co_list = []
        for n in range(num_n):
            co_per_n = []
            for pol, m_end in [("M", num_m + 2), ("E", num_m + 1)]:
                for m in range(1, m_end):
                    alpha = (pol, n, m)
                    co_per_n.append(co(alpha, self.r_ratio))
            self.co_list.append(np.array(co_per_n))

    @property
    def shape(self):
        return "coax"

    @property
    def num_all(self):
        return self.params["num_n"] * (2 * self.params["num_m"])

    def beta2_pec(self, w, n):
        """Return squares of phase constants for a PEC waveguide

        Args:
            w: A complex indicating the angular frequency
            n: A integer indicating the order of the modes.
        Returns:
            h2s: A 1D array indicating squares of phase constants, whose first
                num_mn + 1 elements are for TM-like modes and the rest are for
                TE-like modes.
        """
        w_comp = w.real + 1j * w.imag
        return self.fill(w_comp) * w_comp ** 2 - self.co_list[n] ** 2 / self.r ** 2

    def x(
        self, h2: complex | np.ndarray, w: complex, e1: complex
    ) -> complex | np.ndarray:
        val: complex | np.ndarray = (
            (1 + 1j) * np.sqrt(-0.5j * (e1 * w ** 2 - h2)) * self.ri
        )
        # val: complex | np.ndarray = np.sqrt(e1 * w ** 2 - h2 + 0j) * self.ri
        return val

    def y(
        self, h2: complex | np.ndarray, w: complex, e2: complex
    ) -> complex | np.ndarray:
        val: complex | np.ndarray = (
            (1 - 1j) * np.sqrt(0.5j * (h2 - e2 * w ** 2)) * self.ri
        )
        # val: complex | np.ndarray = np.sqrt(h2 - e2 * w ** 2 + 0j) * self.ri
        return val

    def u(
        self, h2: complex | np.ndarray, w: complex, e1: complex
    ) -> complex | np.ndarray:
        val: complex | np.ndarray = (
            (1 + 1j) * np.sqrt(-0.5j * (e1 * w ** 2 - h2)) * self.r
        )
        # val: complex | np.ndarray = np.sqrt(e1 * w ** 2 - h2 + 0j) * self.r
        return val

    def v(
        self, h2: complex | np.ndarray, w: complex, e2: complex
    ) -> complex | np.ndarray:
        val: complex | np.ndarray = (
            (1 - 1j) * np.sqrt(0.5j * (h2 - e2 * w ** 2)) * self.r
        )
        # val: complex | np.ndarray = np.sqrt(h2 - e2 * w ** 2 + 0j) * self.r
        return val

    def eig_mat(
        self, h2: complex, w: complex, pol: str, n: int, e1: complex, e2: complex
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return matrix of characteristic equation

        Args:
            h2: The square of the phase constant.
            w: The angular frequency
            pol: The polarization
            n: The order of the modes
            e1: The permittivity of the core
            e2: The permittivity of the clad.
        Returns:
            (a, b) (tuple[np.ndarray, np.ndarray]): a: matrix of characteristic equation and b: its defivative.
        """
        h2comp = h2.real + 1j * h2.imag
        w2 = w ** 2
        hew = h2comp / e2 / w2
        ee = e1 / e2
        x = self.x(h2comp, w, e1)
        y = self.y(h2comp, w, e2)
        u = self.u(h2comp, w, e1)
        v = self.v(h2comp, w, e2)

        if u.imag > 0:
            sign_u = 1
        elif u.imag == 0:
            sign_u = 0
        else:
            sign_u = -1

        ph_u = np.exp(1j * sign_u * u.real)

        ju = ssp.jve(n, u) * ph_u
        jpu = -ssp.jve(n + 1, u) * ph_u + n / u * ju
        jppu = -jpu / u - (1 - n ** 2 / u ** 2) * ju

        yu = ssp.yve(n, u) * ph_u
        ypu = -ssp.yve(n + 1, u) * ph_u + n / u * yu
        yppu = -ypu / u - (1 - n ** 2 / u ** 2) * yu

        kv = ssp.kve(n, v)
        kpv = -ssp.kve(n + 1, v) + n / v * kv
        kppv = -kpv / v + (1 + n ** 2 / v ** 2) * kv

        jx = ssp.jve(n, x)
        jpx = -ssp.jve(n + 1, x) + n / x * jx
        jppx = -jpx / x - (1 - n ** 2 / x ** 2) * jx

        yx = ssp.yve(n, x)
        ypx = -ssp.yve(n + 1, x) + n / x * yx
        yppx = -ypx / x - (1 - n ** 2 / x ** 2) * yx

        iy = ssp.ive(n, y)
        ipy = ssp.ive(n + 1, y) + n / y * iy
        ippy = -ipy / y + (1 + n ** 2 / y ** 2) * iy

        du_dh2 = -self.r ** 2 / (2 * u)
        dv_dh2 = self.r ** 2 / (2 * v)
        dx_dh2 = -self.ri ** 2 / (2 * x)
        dy_dh2 = self.ri ** 2 / (2 * y)

        nuv = n * (v / u + u / v)
        dnuv_du = n * (-v / u ** 2 + 1 / v)
        dnuv_dv = n * (-u / v ** 2 + 1 / u)

        nxy = n * (y / x + x / y)
        dnxy_dx = n * (-y / x ** 2 + 1 / y)
        dnxy_dy = n * (-x / y ** 2 + 1 / x)

        a = np.array(
            [
                [
                    jpu * kv * v + kpv * ju * u,
                    ypu * kv * v + kpv * yu * u,
                    nuv * ju * kv,
                    nuv * yu * kv,
                ],
                [
                    jpx / yx * y + ipy / iy * jx / yx * x,
                    ypx / yx * y + ipy / iy * x,
                    nxy * jx / yx,
                    nxy,
                ],
                [
                    hew * nuv * ju * kv,
                    hew * nuv * yu * kv,
                    ee * jpu * kv * v + kpv * ju * u,
                    ee * ypu * kv * v + kpv * yu * u,
                ],
                [
                    hew * nxy * jx / yx,
                    hew * nxy,
                    ee * jpx / yx * y + ipy / iy * jx / yx * x,
                    ee * ypx / yx * y + ipy / iy * x,
                ],
            ]
        )

        da_du = np.array(
            [
                [
                    jppu * kv * v + kpv * (jpu * u + ju) + 1j * sign_u * a[0, 0],
                    yppu * kv * v + kpv * (ypu * u + yu) + 1j * sign_u * a[0, 1],
                    dnuv_du * ju * kv + nuv * jpu * kv + 1j * sign_u * a[0, 2],
                    dnuv_du * yu * kv + nuv * ypu * kv + 1j * sign_u * a[0, 3],
                ],
                [0, 0, 0, 0],
                [
                    hew * (dnuv_du * ju + nuv * jpu) * kv + 1j * sign_u * a[2, 0],
                    hew * (dnuv_du * yu + nuv * ypu) * kv + 1j * sign_u * a[2, 1],
                    ee * jppu * kv * v + kpv * (jpu * u + ju) + 1j * sign_u * a[2, 2],
                    ee * yppu * kv * v + kpv * (ypu * u + yu) + 1j * sign_u * a[2, 3],
                ],
                [0, 0, 0, 0],
            ]
        )

        da_dv = np.array(
            [
                [
                    jpu * (kpv * v + kv) + kppv * ju * u + a[0, 0],
                    ypu * (kpv * v + kv) + kppv * yu * u + a[0, 1],
                    (dnuv_dv * kv + nuv * kpv) * ju + a[0, 2],
                    (dnuv_dv * kv + nuv * kpv) * yu + a[0, 3],
                ],
                [0, 0, 0, 0],
                [
                    hew * (dnuv_dv * kv + nuv * kpv) * ju + a[2, 0],
                    hew * (dnuv_dv * kv + nuv * kpv) * yu + a[2, 1],
                    ee * jpu * (kpv * v + kv) + kppv * ju * u + a[2, 2],
                    ee * ypu * (kpv * v + kv) + kppv * yu * u + a[2, 3],
                ],
                [0, 0, 0, 0],
            ]
        )

        da_dx = np.array(
            [
                [0, 0, 0, 0],
                [
                    (
                        (jppx / yx - jpx * ypx / yx ** 2) * y
                        + ipy / iy * ((jpx / yx - jx * ypx / yx ** 2) * x + jx / yx)
                    ),
                    (yppx / yx - ypx ** 2 / yx ** 2) * y + ipy / iy,
                    dnxy_dx * jx / yx + nxy * jpx / yx - nxy * jx * ypx / yx ** 2,
                    dnxy_dx,
                ],
                [0, 0, 0, 0],
                [
                    hew
                    * (dnxy_dx * jx / yx + nxy * jpx / yx - nxy * jx * ypx / yx ** 2),
                    hew * dnxy_dx,
                    (
                        ee * (jppx / yx - jpx * ypx / yx ** 2) * y
                        + ipy / iy * ((jpx / yx - jx * ypx / yx ** 2) * x + jx / yx)
                    ),
                    ee * (yppx / yx - ypx ** 2 / yx ** 2) * y + ipy / iy,
                ],
            ]
        )

        da_dy = np.array(
            [
                [0, 0, 0, 0],
                [
                    jpx / yx + (ippy / iy - ipy ** 2 / iy ** 2) * jx / yx * x,
                    ypx / yx + (ippy / iy - ipy ** 2 / iy ** 2) * x,
                    dnxy_dy * jx / yx,
                    dnxy_dy,
                ],
                [0, 0, 0, 0],
                [
                    hew * dnxy_dy * jx / yx,
                    hew * dnxy_dy,
                    ee * jpx / yx + (ippy / iy - ipy ** 2 / iy ** 2) * jx / yx * x,
                    ee * ypx / yx + (ippy / iy - ipy ** 2 / iy ** 2) * x,
                ],
            ]
        )

        b = (
            da_du * du_dh2
            + da_dv * dv_dh2
            + da_dx * dx_dh2
            + da_dy * dy_dh2
            + np.array(
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [
                        ee / w2 * nuv * ju * kv,
                        ee / w2 * nuv * yu * kv,
                        0,
                        0,
                    ],
                    [
                        ee / w2 * nxy * jx / yx,
                        ee / w2 * nxy,
                        0,
                        0,
                    ],
                ]
            )
        )
        return a, b

    def eig_eq(
        self,
        h2vec: np.ndarray,
        w: complex,
        pol: str,
        n: int,
        e1: complex,
        e2: complex,
        roots: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the value of the characteristic equation

        Args:
            h2vec: The real and imaginary parts of the square of propagation constant.
            w: The angular frequency
            pol: The polarization
            n: The order of the modes
            e1: The permittivity of the core
            e2: The permittivity of the clad.
        Returns:
            val: A complex indicating the left-hand value of the characteristic
                equation.
        """
        num = len(roots)
        h2 = h2vec[0] + h2vec[1] * 1j
        a, b = self.eig_mat(h2, w, pol, n, e1, e2)
        if n == 0:
            if pol == "E":
                f = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
                fp = eig_mat_utils.deriv_det2_cython(
                    np.ascontiguousarray(a[:2, :2]), np.ascontiguousarray(b[:2, :2])
                )
            else:
                f = a[2, 2] * a[3, 3] - a[2, 3] * a[3, 2]
                fp = eig_mat_utils.deriv_det2_cython(
                    np.ascontiguousarray(a[2:, 2:]), np.ascontiguousarray(b[2:, 2:])
                )
        else:
            f = np.linalg.det(a)
            fp = eig_mat_utils.deriv_det4_cython(a, b)
        denom = 1.0
        dd = 0.0
        tanhs = np.tanh(h2 - roots)
        for i in range(num):
            denom *= tanhs[i]
            ddi = (tanhs[i] ** 2 - 1) / tanhs[i] ** 2
            for j in range(num):
                if j != i:
                    ddi /= tanhs[j]
            dd += ddi
        fp = fp / denom + f * dd
        f /= denom
        return np.array([f.real, f.imag]), np.array(
            [[fp.real, -fp.imag], [fp.imag, fp.real]]
        )

    def beta2(
        self, w: complex, n: int, e1: complex, e2: complex, xis: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return roots and convergences of the characteristic equation

        Args:
            w (complex): Angular frequency.
            n (int): Order of the mode
            e1 (complex): Permittivity of tha core.
            e2 (complex): Permittivity of tha clad.
            xis (np.ndarray): Initial approximations for the roots
                whose number of elements is 2*num_m+1.
        Returns:
            xs: A 1D array indicating the roots, whose length is 2*num_m+1.
            success: A 1D array indicating the convergence information for xs.
        """
        if self.clad.label == "PEC":
            xs = self.beta2_pec(w, n)
            return xs, np.ones_like(xs, dtype=bool)
        num_m = self.params["num_m"]
        roots: list[complex] = []
        vals = []
        success: list[bool] = []

        for i, xi in enumerate(xis):
            if i < num_m + 1:
                pol = "M"
            else:
                pol = "E"
                if n == 0 and i == num_m + 1:
                    roots = []
            # result = minimize(
            #     coax_utils.eig_eq_for_min,
            #     np.array([xi.real, xi.imag]),
            #     args=(w, pol, n, e1, e2, self.r, self.ri, np.array(roots, dtype=complex)),
            #     jac=True,
            # )
            result = root(
                coax_utils.eig_eq_with_jac,
                np.array([xi.real, xi.imag]),
                args=(
                    w,
                    pol,
                    n,
                    e1,
                    e2,
                    self.r,
                    self.ri,
                    np.array(roots, dtype=complex),
                ),
                jac=True,
                method="hybr",
                options={"xtol": 1.0e-9},
            )
            x = result.x[0] + result.x[1] * 1j
            v = self.v(x, w, e2)
            if result.success:
                roots.append(x)
            if v.real > 0.0:
                success.append(result.success)
            else:
                success.append(False)
            vals.append(x)
        return np.array(vals), np.array(success)

    @staticmethod
    def beta_from_beta2(x):
        return (1 + 1j) * np.sqrt(-0.5j * x)

    def beta2_w_min(self, n):
        """Return roots and convergences of the characteristic equation at
            the lowest angular frequency, ws[0].

        Args:
            n: A integer indicating the order of the mode
        Returns:
            xs: A 1D array indicating the roots, whose length is 2*num_m+1.
            success: A 1D array indicating the convergence information for xs.
        """
        if self.clad.label == "PEC":
            xs = self.beta2_pec(self.ws[0], n)
            success = np.ones_like(xs, dtype=bool)
            return xs, success
        w_0 = 2 * np.pi / 100.0
        e1 = self.fill(w_0)
        e2_0 = self.clad(w_0) * 1000
        de2 = (self.clad(w_0) - e2_0) / 1000
        xs, success = self.beta2(w_0, n, e1, e2_0, self.beta2_pec(w_0, n))
        xs0 = xs1 = xs
        success = np.ones_like(xs0, dtype=bool)
        for i in range(1001):
            e2 = e2_0 + de2 * i
            # if i % 100 == 0 and n == 0:
            #     print(e2)
            #     print(xs1)
            xis = 2 * xs1 - xs0
            xs, success = self.beta2(w_0, n, e1, e2, xis)
            xs0 = xs1
            xs1 = xs
        xs0 = xs1
        dw = (self.ws[0] - w_0) / 1000
        for i in range(1001):
            w = w_0 + dw * i
            e1 = self.fill(w)
            e2 = self.clad(w)
            xis = 2 * xs1 - xs0
            xs, success = self.beta2(w, n, e1, e2, xis)
            xs0 = xs1
            xs1 = xs
        return xs, success

    def betas_convs(self, xs_success_list: list) -> tuple[dict, dict]:
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
            xs = np.where(success, xs, xis)
            xs_array[iwr, iwi] = xs
            success_array[iwr, iwi] = success
            xis = xs
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
                xs = np.where(success, xs, xis)
                xs_array[iwr, iwi] = xs
                success_array[iwr, iwi] = success
        return xs_array, success_array


class SamplesLowLoss(Samples):
    """A class defining samples of phase constants of coaxial waveguide
    modes in a virtual low-loss clad waveguide by subclassing the Samples class.

    Attributes:
        fill: An instance of Material class for the core
        clad: An instance of Material class for the clad
        size: A float indicating the size of core [um].
        size2: A float indicating the optional size of core [um].
        params: A dict whose keys and values are as follows:
            'wl_max': A float indicating the maximum wavelength [um]
            'wl_min': A float indicating the minimum wavelength [um]
            'wl_imag': A float indicating the minimum value of
                abs(c / f_imag) [um] where f_imag is the imaginary part of
                the frequency.
            'dw': A float indicating frequency interval
                [rad * c / 1um]=[2.99792458e14 rad / s].
            'num_n': An integer indicating the number of orders of modes.
            'num_m': An integer indicating the number of modes in each
                order and polarization.
        ws: A 1D array indicating the real part of the angular frequencies
            to be calculated [rad (c / 1um)]=[2.99792458e14 rad / s].
        wis: A 1D array indicating the imaginary part of the angular
            frequencies to be calculated [rad * (c / 1um)].
        ri: A float indicating inner radius [um].
        r: A float indicating outer radius [um].
        co_list: A list indicating cutoffs for PEC waveguide.
    """

    def __init__(self, size: float, fill: dict, clad: dict, params: dict, size2):
        """Init Samples class.

        size (float): The radius of the cross section [um]
        fill (dict): Parameters for riip.Material class for the core
        clad (dict): Parameters for riip.Material class for the clad
        params (dict): Keys and values are as follows:
            'wl_max' (float): The maximum wavelength [um].
                Defaults to 5.0.
            'wl_min' (float): The minimum wavelength [um].
                Defaults to 0.4.
            'wl_imag' (float): The minimum value of abs(c / f_imag) [um]
                where f_imag is the imaginary part of the frequency.
                Defaults to 5.0.
            'dw' (float): The frequency interval [rad c / 1um]=[2.99792458e14 rad / s].
                Defaults to 1 / 64.
            'num_n' (int): The number of orders of modes.
            'num_m' (int): The number of modes in each order and polarization.
        """
        super().__init__(size, fill, clad, params, size2)

    def betas_convs(self, xs_success_list):
        num_iwr = len(self.ws)
        num_iwi = len(self.wis)
        num_n = self.params["num_n"]
        num_m = self.params["num_m"]
        betas = {}
        convs = {}
        for n in range(num_n):
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
        for iwr in range(num_iwr):
            for iwi in range(num_iwi):
                j = iwr * num_iwi + iwi
                w = self.ws[iwr] + 1j * self.wis[iwi]
                e2 = self.clad(w)
                for n in range(num_n):
                    for i in range(num_m + 1):
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


@ray.remote
class SamplesForRay(Samples):
    """A derived class in order to create ray actor."""

    def __init__(self, size: float, fill: dict, clad: dict, params: dict, size2: float):
        super().__init__(size, fill, clad, params, size2)

    def task(self, n: int):
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
        return super().__call__(n)


@ray.remote
class SamplesLowLossForRay(SamplesLowLoss):
    """A derived class in order to create ray actor."""

    def __init__(self, size: float, fill: dict, clad: dict, params: dict, size2: float):
        super().__init__(size, fill, clad, params, size2)

    def task(self, arg: tuple[int, int, list[np.ndarray]]):
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
            for i in range(1, 8):
                self.clad.im_factor = 0.5 ** i
                if i == 7 or self.clad.im_factor < im_factor:
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
