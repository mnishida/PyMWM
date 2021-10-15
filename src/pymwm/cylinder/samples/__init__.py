from __future__ import annotations

import cmath

import numpy as np
import ray
import riip
import scipy.special as ssp
from scipy.optimize import root

from pymwm.utils import cylinder_utils, eig_mat_utils
from pymwm.waveguide import Sampling


class Samples(Sampling):
    """A class defining samples of phase constants of cylindrical waveguide
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
        r: A float indicating the radius of the circular cross section [um].
    """

    def __init__(self, size: float, fill: dict, clad: dict, params: dict):
        """Init Samples class.

        Args:
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
        super().__init__(size, fill, clad, params)
        self.r = size

    @property
    def shape(self):
        return "cylinder"

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
        w_comp = w.real + 1j * w.imag
        # The number of TM-like modes for each order is taken lager than
        # the number of TE-like modes since TM-like modes are almost identical
        # to those for PEC waveguide and can be calculated easily.
        # Dividing function by (x - x0) where x0 is already-found root
        # makes it easier to find new roots.
        num_m = self.params["num_m"]
        chi = ssp.jn_zeros(n, num_m + 1)
        h2s_mag = self.fill(w_comp) * w_comp ** 2 - chi ** 2 / self.r ** 2
        chi = ssp.jnp_zeros(n, num_m)
        h2s_elec = self.fill(w_comp) * w_comp ** 2 - chi ** 2 / self.r ** 2
        h2s = np.hstack((h2s_mag, h2s_elec))
        return h2s

    def u(
        self, h2: complex | np.ndarray, w: complex, e1: complex
    ) -> complex | np.ndarray:
        # val: complex | np.ndarray = 1j * np.sqrt(-e1 * w ** 2 + h2) * self.r
        val: complex | np.ndarray = (
            (1 + 1j) * np.sqrt(-0.5j * (e1 * w ** 2 - h2)) * self.r
        )
        # val: complex | np.ndarray = np.sqrt(e1 * w ** 2 - h2) * self.r
        return val

    def v(
        self, h2: complex | np.ndarray, w: complex, e2: complex
    ) -> complex | np.ndarray:
        val: complex | np.ndarray = (
            (1 - 1j) * np.sqrt(0.5j * (-e2 * w ** 2 + h2)) * self.r
        )
        # val: complex | np.ndarray = np.sqrt(-e2 * w ** 2 + h2) * self.r
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
        u = self.u(h2comp, w, e1)
        v = self.v(h2comp, w, e2)
        if u.imag > 0:
            sign = 1
        elif u.imag == 0:
            sign = 0
        else:
            sign = -1
        ph = np.exp(1j * sign * u.real)
        ju = ssp.jve(n, u) * ph
        jpu = -ssp.jve(n + 1, u) * ph + n * ju / u
        jppu = -jpu / u - (1 - n ** 2 / u ** 2) * ju
        kv = ssp.kve(n, v)
        kpv = -ssp.kve(n + 1, v) + n * kv / v
        kppv = -kpv / v + (1 + n ** 2 / v ** 2) * kv
        te = jpu * kv * v + kpv * ju * u
        tm = ee * jpu * kv * v + kpv * ju * u

        # print(f"{u=}")
        # print(f"{v=}")
        # print(f"{ju=}")
        # print(f"{jpu=}")
        # print(f"{kv=}")
        # print(f"{kpv=}")

        du_dh2 = -self.r ** 2 / (2 * u)
        dv_dh2 = self.r ** 2 / (2 * v)

        dte_du = jppu * kv * v + kpv * (jpu * u + ju) + 1j * sign * te
        dte_dv = jpu * (kpv * v + kv) + kppv * ju * u + te
        dte_dh2 = dte_du * du_dh2 + dte_dv * dv_dh2

        dtm_du = ee * (jppu * kv * v) + kpv * (jpu * u + ju) + 1j * sign * tm
        dtm_dv = ee * jpu * (kpv * v + kv) + kppv * ju * u + tm
        dtm_dh2 = dtm_du * du_dh2 + dtm_dv * dv_dh2

        if n == 0:
            if pol == "M":
                return np.array([[tm]]), np.array([[dtm_dh2]])
            else:
                return np.array([[te]]), np.array([[dte_dh2]])

        nuv = n * (v / u + u / v) * ju * kv
        dnuv_du = (
            n * (-v / u ** 2 + 1 / v) * ju * kv
            + n * (v / u + u / v) * (jpu + 1j * sign * ju) * kv
        )
        dnuv_dv = n * (-u / v ** 2 + 1 / u) * ju * kv + n * (v / u + u / v) * ju * (
            kpv + kv
        )
        dnuv_dh2 = dnuv_du * du_dh2 + dnuv_dv * dv_dh2
        return (
            np.array([[te, nuv], [hew * nuv, tm]]),
            np.array([[dte_dh2, dnuv_dh2], [hew * dnuv_dh2 + nuv / e2 / w2, dtm_dh2]]),
        )

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
        h2 = h2vec[0] + h2vec[1] * 1j
        a, b = self.eig_mat(h2, w, pol, n, e1, e2)
        num = len(roots)
        if n == 0:
            f: complex = a[0, 0]
            fp: complex = b[0, 0]
        else:
            f = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
            fp = eig_mat_utils.deriv_det2_cython(a, b)
        denom = 1.0
        dd = 0.0
        z = np.where((h2 - roots).real >= 0, h2 - roots, roots - h2)
        val = np.exp(-2 * z)
        tanhs = np.where(
            (h2 - roots).real >= 0, (1 - val) / (1 + val), (val - 1) / (val + 1)
        )
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
            [[fp.real, fp.imag], [-fp.imag, fp.real]]
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
            result = root(
                cylinder_utils.eig_eq_cython,
                np.array([xi.real, xi.imag]),
                args=(w, pol, n, e1, e2, self.r, np.array(roots, dtype=complex)),
                jac=True,
                method="hybr",
                options={"xtol": 1.0e-9, "col_deriv": True},
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
        w_0 = 2 * np.pi / 100
        e1 = self.fill(w_0)
        e2_0 = self.clad(w_0) * 1000
        de2 = (self.clad(w_0) - e2_0) / 1000
        xs, success = self.beta2(w_0, n, e1, e2_0, self.beta2_pec(w_0, n))
        xs0 = xs1 = xs
        success = np.ones_like(xs0, dtype=bool)
        for i in range(1001):
            e2 = e2_0 + de2 * i
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
                    for i in range(num_m + 1):
                        x = xs_array[iwr, iwi][i]
                        betas[("M", n, i + 1)][iwr, iwi] = self.beta_from_beta2(x)
                        convs[("M", n, i + 1)][iwr, iwi] = success_array[iwr, iwi][i]
                    for i in range(num_m):
                        x = xs_array[iwr, iwi][i + num_m + 1]
                        betas[("E", n, i + 1)][iwr, iwi] = self.beta_from_beta2(x)
                        convs[("E", n, i + 1)][iwr, iwi] = success_array[iwr, iwi][
                            i + num_m + 1
                        ]
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

    def wr_sampling(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        num_m = self.params["num_m"]
        xs_array = np.zeros((len(self.ws), 2 * num_m + 1), dtype=complex)
        success_array = np.zeros((len(self.ws), 2 * num_m + 1), dtype=bool)
        iwr = 0
        xis, success = self.beta2_w_min(n)
        xs_array[iwr] = xis
        success_array[iwr] = success
        xs0 = xs1 = xis
        for iwr in range(1, len(self.ws)):
            w = self.ws[iwr]
            e1 = self.fill(w)
            e2 = self.clad(w)
            xis = 2 * xs1 - xs0
            xs, success = self.beta2(w, n, e1, e2, xis)
            xs = np.where(success, xs, xis)
            xs_array[iwr] = xs
            success_array[iwr] = success
            xs0 = xs1
            xs1 = xs
        return xs_array, success_array

    def wi_sampling(
        self, args: tuple[int, int, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        n, iwr, xis0 = args
        num_m = self.params["num_m"]
        xs_array = np.zeros((len(self.wis), 2 * num_m + 1), dtype=complex)
        success_array = np.zeros((len(self.wis), 2 * num_m + 1), dtype=bool)
        wr = self.ws[iwr]
        xs0 = xs1 = xis0
        for iwi in range(len(self.wis)):
            wi = self.wis[iwi]
            w = wr + 1j * wi
            e1 = self.fill(w)
            e2 = self.clad(w)
            xis = 2 * xs1 - xs0
            xs, success = self.beta2(w, n, e1, e2, xis)
            xs = np.where(success, xs, xis)
            xs_array[iwi] = xs
            success_array[iwi] = success
            xs0 = xs1
            xs1 = xs
        return xs_array, success_array


class SamplesLowLoss(Samples):
    """A class defining samples of phase constants of cylindrical waveguide
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
        r: A float indicating the radius of the circular cross section [um].
    """

    def __init__(self, size: float, fill: dict, clad: dict, params: dict):
        """Init SamplesLowLoss class.

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
        super().__init__(size, fill, clad, params)

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

    def __init__(self, size: float, fill: dict, clad: dict, params: dict):
        super().__init__(size, fill, clad, params)


@ray.remote
class SamplesLowLossForRay(SamplesLowLoss):
    """A derived class in order to create ray actor."""

    def __init__(self, size: float, fill: dict, clad: dict, params: dict):
        super().__init__(size, fill, clad, params)

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
