from __future__ import annotations

import cmath
from logging import getLogger

import numpy as np
import ray
import riip

from pymwm.utils.slit_utils import func_cython
from pymwm.waveguide import Sampling

logger = getLogger(__package__)


class Samples(Sampling):
    """A class defining samples of phase constants of slit waveguide modes.

    Attributes:
        r: A float indicating the radius of the circular cross section [um].
    """

    def __init__(self, size: float, fill: dict, clad: dict, params: dict):
        """Init Samples class.

        Args:
            size (float): The width of slit [um].
            fill (dict): Parameters for riip.Material class for the core.
            clad (dict): Parameters for riip.Material class for the clad.
            params (dict): Keys and values are as follows:
                'wl_max' (float): The maximum wavelength [um].
                    Defaults to 5.0.
                'wl_min' (float): The minimum wavelength [um].
                    Defaults to 0.4.
                'wl_imag' (float): The minimum value of
                    abs(c / f_imag) [um] where f_imag is the imaginary part of
                    the frequency. Defaults to 5.0.
                'dw' (float): The frequency interval [rad c / 1um]=[2.99792458e14 rad / s].
                    Defaults to 1 / 64.
                'num_n (int)': The number of orders of modes.
                'num_m' (int): The number of modes in each order and polarization
                    (= 1 in the slit case).
        """
        num_m = params.setdefault("num_m", 1)
        if num_m != 1:
            logger.warning(
                "num_m must be 1 if shape is slit." + "The set value is ignored."
            )
            params["num_m"] = 1
        super().__init__(size, fill, clad, params)
        self.r = size

    @property
    def shape(self):
        return "slit"

    @property
    def num_all(self):
        return 2 * self.params["num_n"]

    def beta2_pec(self, w: complex, parity: str, num_n: int) -> np.ndarray:
        """Return squares of phase constants for a PEC waveguide

        Args:
            w (complex): Angular frequency
            parity ("even" or "odd"): "even" ("odd") if even (odd) numbers in the list of n are evaluated.
            num_n (int): Number of the modes.
        Returns:
            h2s (np.ndarray): The squares of phase constants, whose first
                element is for TM mode and the rest is for both TE  and TM modes.
        """
        w_comp = w.real + 1j * w.imag
        ns_all = list(range(num_n))
        if parity == "even":
            ns = np.array(ns_all[::2])
        else:
            ns = np.array(ns_all[1::2])
        h2: np.ndarray = self.fill(w_comp) * w_comp ** 2 - (ns * np.pi / self.r) ** 2
        return h2

    def u(self, h2: complex, w: complex, e1: complex) -> complex:
        # return cmath.sqrt(e1 * w ** 2 - h2) * self.r / 2
        return (1 + 1j) * cmath.sqrt(-0.5j * (e1 * w ** 2 - h2)) * self.r / 2

    def v(self, h2: complex, w: complex, e2: complex) -> complex:
        # This definition is very important!!
        # Other definitions can not give good results in some cases
        return (1 - 1j) * cmath.sqrt(0.5j * (-e2 * w ** 2 + h2)) * self.r / 2

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
        u = self.u(h2, w, e1)
        v = self.v(h2, w, e2)
        if pol == "E":
            if n % 2 == 0:
                return u / v + cmath.tan(u)
            else:
                return u / v - 1 / cmath.tan(u)
        else:
            if n % 2 == 0:
                return u * cmath.tan(u) - (e1 * v) / e2
            else:
                return u / cmath.tan(u) + (e1 * v) / e2

    def beta2(
        self,
        w: complex,
        pol: str,
        parity: str,
        num_n: int,
        e1: complex,
        e2: complex,
        xis: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return roots and convergences of the characteristic equation

        Args:
            w (complex): Angular frequency.
            pol (str): 'E' or 'M' indicating the polarization.
            parity ("even" or "odd"): "even" ("odd") if even (odd) numbers in the list of n are evaluated.
            num_n (int): The number of modes.
            e1 (complex): Permittivity of tha core.
            e2 (complex): Permittivity of tha clad.
            xis (np.ndarray[complex]): Initial approximations for the roots whose number of
               elements is 2.
        Returns:
            xs: The roots, whose length is 2.
            success: The convergence information for xs.
        """
        if self.clad.label == "PEC":
            xs = self.beta2_pec(w, parity, num_n)
            s = np.ones_like(xs, dtype=bool)
            if parity == "even" and pol == "E":
                s[0] = False
            return xs, s
        from scipy.optimize import root

        roots: list[complex] = []
        vals: list[complex] = []
        success: list[bool] = []
        ns_all = list(range(num_n))
        if parity == "even":
            ns = ns_all[::2]
        else:
            ns = ns_all[1::2]

        for i_n, n in enumerate(ns):
            xi = xis[i_n]
            if pol == "E" and n == 0:
                vals.append(xi)
                success.append(False)
                continue
            result = root(
                func_cython,
                np.array([xi.real, xi.imag]),
                args=(w, pol, n, e1, e2, self.r, np.array(roots, dtype=complex)),
                method="hybr",
                options={"xtol": 1.0e-9},
            )
            x = result.x[0] + result.x[1] * 1j
            if result.success:
                roots.append(x)

            v = self.v(x, w, e2)
            if v.real > 0.0:
                success.append(result.success)
            else:
                success.append(False)
            vals.append(x)
        return np.array(vals), np.array(success)

    @staticmethod
    def beta_from_beta2(x: np.ndarray):
        return (1 + 1j) * np.sqrt(-0.5j * x)

    def beta2_w_min(
        self, pol: str, parity: str, num_n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return roots and convergences of the characteristic equation at
            the lowest angular frequency, ws[0].

        Args:
            pol (str): 'E' or 'M' indicating the polarization.
            parity ("even" or "odd"): "even" ("odd") if even (odd) numbers in the list of n are evaluated.
            num_n (int): An integer indicating the number of modes.
        Returns:
            xs (np.ndarray): A 1D array indicating the roots, whose length is 2.
            success (np.ndarray): A 1D array indicating the convergence information for xs.
        """
        if self.clad.label == "PEC":
            xs = self.beta2_pec(self.ws[0], parity, num_n)
            success = np.ones_like(xs, dtype=bool)
            if parity == "even" and pol == "E":
                success[0] = False
            return xs, success
        w_0 = 0.1
        e1 = self.fill(w_0)
        e2_0 = -5.0e5 + self.clad(w_0).imag * 1j
        de2 = (self.clad(w_0) - e2_0) / 1000
        xis = xs = self.beta2_pec(w_0, parity, num_n)
        success = np.ones_like(xs, dtype=bool)
        for i in range(1001):
            e2 = e2_0 + de2 * i
            xs, success = self.beta2(w_0, pol, parity, num_n, e1, e2, xis)
            for _, ok in enumerate(success):
                if not ok:
                    xs[_] = xis[_]
            xis = xs
        dw = (self.ws[0] - w_0) / 1000
        for i in range(1001):
            w = w_0 + dw * i
            e1 = self.fill(w)
            e2 = self.clad(w)
            xs, success = self.beta2(w, pol, parity, num_n, e1, e2, xis)
            for _, ok in enumerate(success):
                if not ok:
                    xs[_] = xis[_]
            xis = xs
        return xs, success

    def betas_convs(self, xs_success_list):
        betas = {}
        convs = {}
        for i_pol, pol in enumerate(["M", "E"]):
            xs_array, success_array = xs_success_list[i_pol]
            num_n = xs_array.shape[2]
            for n in range(num_n):
                betas[(pol, n, 1)] = np.zeros(
                    (len(self.ws), len(self.wis)), dtype=complex
                )
                convs[(pol, n, 1)] = np.zeros((len(self.ws), len(self.wis)), dtype=bool)
            for iwi in range(len(self.wis)):
                for iwr in range(len(self.ws)):
                    w = self.ws[iwr] + 1j * self.wis[iwi]
                    e2 = self.clad(w)
                    for n in range(num_n):
                        x = xs_array[iwr, iwi][n]
                        v = self.v(x, w, e2)
                        betas[(pol, n, 1)][iwr, iwi] = self.beta_from_beta2(x)
                        convs[(pol, n, 1)][iwr, iwi] = (
                            success_array[iwr, iwi][n]
                            if v.real > abs(v.imag)
                            else False
                        )
        return betas, convs

    def __call__(self, arg: tuple[str, str, int]) -> tuple[np.ndarray, np.ndarray]:
        """Return a dict of the roots of the characteristic equation

        Args:
            arg: (pol, parity, num_n)
                pol ("E" or "M"): Polarization.
                parity ("even" or "odd"): "even" ("odd") if even (odd) numbers in the list of n are evaluated.
                num_n (int): The number of modes.
        Returns:
            betas: A dict containing arrays of roots, whose key is as follows:
                (pol, n, m):
                    pol: 'E' or 'M' indicating the polarization.
                    n: An integer indicating the order of the mode.
                    m: An integer indicating the ordinal of the mode in the
                        same order.
            convs: A dict containing the convergence information for betas,
                whose key is the same as above.
        """
        pol, parity, num_n = arg
        num_ws = len(self.ws)
        ns = list(range(num_n))
        if parity == "even":
            num = len(ns[::2])
        else:
            num = len(ns[1::2])
        xs_array = np.zeros((num_ws, len(self.wis), num), dtype=complex)
        success_array = np.zeros((num_ws, len(self.wis), num), dtype=bool)
        iwr = iwi = 0
        wi = self.wis[iwi]
        xis, success = self.beta2_w_min(pol, parity, num_n)
        xs_array[iwr, iwi] = xis
        success_array[iwr, iwi] = success
        for iwr in range(1, len(self.ws)):
            wr = self.ws[iwr]
            w = wr + 1j * wi
            e1 = self.fill(w)
            e2 = self.clad(w)
            xs, success = self.beta2(w, pol, parity, num_n, e1, e2, xis)
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
                xs, success = self.beta2(w, pol, parity, num_n, e1, e2, xis)
                xs = np.where(success, xs, xis)
                xs_array[iwr, iwi] = xs
                success_array[iwr, iwi] = success
        return xs_array, success_array


class SamplesLowLoss(Samples):
    """A class defining samples of phase constants of cylindrical waveguide
    modes in a virtual low-loss clad waveguide by subclassing the SlitSamples
    class.

    Attributes:
        fill: An instance of Material class for the core
        clad: An instance of Material class for the clad
        r: A float indicating the width of the slit [um].
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

    def __init__(self, size: float, fill: dict, clad: dict, params: dict):
        """Init Samples class.

        Args:
            size: A float indicating the width of the slit [um].
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
        """
        super(SamplesLowLoss, self).__init__(size, fill, clad, params)

    def betas_convs(self, xs_success_list):
        num_iwr = len(self.ws)
        num_iwi = len(self.wis)
        num_n = self.params["num_n"]
        ns = list(range(num_n))
        ns_e = ns[::2]
        ns_o = ns[1::2]
        betas = {}
        convs = {}
        for n in ns:
            betas[("M", n, 1)] = np.zeros((len(self.ws), len(self.wis)), dtype=complex)
            convs[("M", n, 1)] = np.zeros((len(self.ws), len(self.wis)), dtype=bool)
            betas[("E", n, 1)] = np.zeros((len(self.ws), len(self.wis)), dtype=complex)
            convs[("E", n, 1)] = np.zeros((len(self.ws), len(self.wis)), dtype=bool)
        for iwr in range(num_iwr):
            for iwi in range(num_iwi):
                j = iwr * num_iwi + iwi
                w = self.ws[iwr] + 1j * self.wis[iwi]
                e2 = self.clad(w)
                for n in ns_e:
                    x = xs_success_list[j][0][0][n]
                    v = self.v(x, w, e2)
                    betas[("M", n, 1)][iwr, iwi] = self.beta_from_beta2(x)
                    convs[("M", n, 1)][iwr, iwi] = (
                        xs_success_list[j][1][0][n] if v.real > abs(v.imag) else False
                    )
                    x = xs_success_list[j][0][2][n]
                    v = self.v(x, w, e2)
                    betas[("E", n, 1)][iwr, iwi] = self.beta_from_beta2(x)
                    convs[("E", n, 1)][iwr, iwi] = (
                        xs_success_list[j][1][1][n] if v.real > abs(v.imag) else False
                    )
                for n in ns_o:
                    x = xs_success_list[j][0][1][n]
                    v = self.v(x, w, e2)
                    betas[("M", n, 1)][iwr, iwi] = self.beta_from_beta2(x)
                    convs[("M", n, 1)][iwr, iwi] = (
                        xs_success_list[j][1][1][n] if v.real > abs(v.imag) else False
                    )
                    x = xs_success_list[j][0][3][n]
                    v = self.v(x, w, e2)
                    betas[("E", n, 1)][iwr, iwi] = self.beta_from_beta2(x)
                    convs[("E", n, 1)][iwr, iwi] = (
                        xs_success_list[j][1][1][n] if v.real > abs(v.imag) else False
                    )
        return betas, convs


@ray.remote
class SamplesForRay(Samples):
    """A derived class in order to create ray actor."""

    def __init__(self, size: float, fill: dict, clad: dict, params: dict):
        super().__init__(size, fill, clad, params)

    def task(self, arg: tuple[str, str, int]) -> tuple[np.ndarray, np.ndarray]:
        """Return a dict of the roots of the characteristic equation

        Args:
            arg: (pol, parity, num_n)
                pol ("E" or "M"): Polarization.
                parity ("even" or "odd"): "even" ("odd") if even (odd) numbers in the list of n are evaluated.
                num_n (int): The number of modes.
        Returns:
            betas: A dict containing arrays of roots, whose key is as follows:
                (pol, n, m):
                    pol: 'E' or 'M' indicating the polarization.
                    n: An integer indicating the order of the mode.
                    m: An integer indicating the ordinal of the mode in the
                        same order.
            convs: A dict containing the convergence information for betas,
                whose key is the same as above.
        """
        return super().__call__(arg)


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
                xis_list: The initial guess of roots whose length is num_n
        Returns:
            xs_list: A list of num_n 1D arrays indicating the roots, whose
                length is num_n
            success_list: A list of num_n 1D arrays indicating the convergence
                information for xs, whose length is num_n
        """
        num_n = self.params["num_n"]
        iwr, iwi, xis_list = arg
        im_factor = self.clad.im_factor
        self.clad.im_factor = 1.0
        wr = self.ws[iwr]
        wi = self.wis[iwi]
        w = wr + 1j * wi
        e1 = self.fill(w)
        xs_list = []
        success_list = []
        for i_pp, x0s in enumerate(xis_list):
            if i_pp == 0:
                pol = "M"
                parity = "even"
            elif i_pp == 1:
                pol = "M"
                parity = "odd"
            elif i_pp == 2:
                pol = "E"
                parity = "even"
            else:
                pol = "E"
                parity = "odd"
            xis = xs = x0s
            success = np.ones_like(xs, dtype=bool)
            for i in range(1, 16):
                self.clad.im_factor = 0.7 ** i
                if i == 15 or self.clad.im_factor < im_factor:
                    self.clad.im_factor = im_factor
                e2 = self.clad(w)
                xs, success = self.beta2(w, pol, parity, num_n, e1, e2, xis)
                for _, ok in enumerate(success):
                    if not ok:
                        xs[_] = xis[_]
                xis = xs
            xs_list.append(xs)
            success_list.append(success)
        return xs_list, success_list
