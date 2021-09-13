from __future__ import annotations

from logging import getLogger

import numpy as np
import ray
import riip

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

    def beta2_pec(self, w, num_n):
        """Return squares of phase constants for a PEC waveguide

        Args:
            w: A complex indicating the angular frequency
            num_n: A integer indicating the number of the modes.
        Returns:
            h2s: A 1D array indicating squares of phase constants, whose first
                element is for TM mode and the rest is for TE mode.
        """
        w_comp = w.real + 1j * w.imag
        ns = np.arange(num_n)
        h2 = self.fill(w_comp) * w_comp ** 2 - (ns * np.pi / self.r) ** 2
        return h2

    def u(self, h2: complex, w: complex, e1: complex) -> complex:
        # return np.sqrt(e1 * w ** 2 - h2) * self.r / 2
        return (1 + 1j) * np.sqrt(-0.5j * (e1 * w ** 2 - h2)) * self.r / 2

    def v(self, h2: complex, w: complex, e2: complex) -> complex:
        # This definition is very important!!
        # Other definitions can not give good results in some cases
        return (1 - 1j) * np.sqrt(0.5j * (-e2 * w ** 2 + h2)) * self.r / 2

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
        h2comp = h2.real + 1j * h2.imag
        u = self.u(h2comp, w, e1)
        v = self.v(h2comp, w, e2)
        if pol == "E":
            if n % 2 == 0:
                return u / v + np.tan(u)
            else:
                return u / v - 1 / np.tan(u)
        else:
            if n % 2 == 0:
                return u * np.tan(u) - (e1 * v) / e2
            else:
                return u / np.tan(u) + (e1 * v) / e2

    def beta2(
        self,
        w: complex,
        pol: str,
        num_n: int,
        e1: complex,
        e2: complex,
        xis: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return roots and convergences of the characteristic equation

        Args:
            w: The angular frequency.
            pol: 'E' or 'M' indicating the polarization.
            num_n: The number of modes.
            e1: The permittivity of tha core.
            e2: The permittivity of tha clad.
            xis: The initial approximations for the roots whose number of
               elements is 2.
        Returns:
            xs: The roots, whose length is 2.
            success: The convergence information for xs.
        """
        if self.clad(w).real < -1e7:
            xs = self.beta2_pec(w, num_n)
            success = np.ones(num_n, dtype=bool)
            if pol == "E":
                success[0] = False
            return xs, success
        from scipy.optimize import root

        roots = []
        vals = []
        success = []
        for n in range(num_n):
            xi = xis[n]
            if pol == "E" and n == 0:
                vals.append(xi)
                success.append(False)
                continue

            def func(h2vec):
                h2 = h2vec[0] + h2vec[1] * 1j
                f = self.eig_eq(h2, w, pol, n, e1, e2)
                prod_denom = 1.0
                for h2_0 in roots:
                    denom = h2 - h2_0
                    while abs(denom) < 1e-14:
                        denom += 1.0e-14
                    prod_denom *= 1.0 / denom
                f *= prod_denom
                f_array = np.array([f.real, f.imag])
                return f_array

            result = root(
                func,
                np.array([xi.real, xi.imag]),
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
        return np.array(vals), success

    @staticmethod
    def beta_from_beta2(x: np.ndarray):
        return (1 + 1j) * np.sqrt(-0.5j * x)

    def func(self, uv, args):
        """Return the values of characteristic equations.

        Args:
            uv: A 1D array of 4 floats indicating (Re[u], Im[u], Re[v], Im[v]).
            args: A tuple (w, pol, n, e1, e2), where w indicates the angular
                frequency, pol indicates the polarization, n indicates
                the order of the modes, e1 indicates the permittivity of
                the core, and e2 indicates the permittivity of the clad.
        Returns:
            vals: A 1D array of 4 floats indicating the left-hand value of the
                characteristic equations.
        """
        w, pol, n, e1, e2 = args
        w2 = (self.r * w / 2) ** 2 * (e1 - e2)
        ur, ui, vr, vi = uv
        u = ur + 1j * ui
        v = vr + 1j * vi
        val2 = u ** 2 + v ** 2 - w2
        if pol == "E":
            if n % 2 == 0:
                val = u / np.tan(u) + v
            else:
                val = u * np.tan(u) - v
        else:
            if n % 2 == 0:
                val = u * np.tan(u) - e1 * (v / e2)
            else:
                val = u / np.tan(u) + e1 * (v / e2)
        return np.array([val.real, val.imag, val2.real, val2.imag])

    @staticmethod
    def jac(uv, args):
        """Return the Jacobian of characteristic equations.

        Args:
            uv: A 1D array of 4 floats indicating (Re[u], Im[u], Re[v], Im[v]).
            args: A tuple (w, pol, n, e1, e2), where w indicates the angular
                frequency, pol indicates the polarization, n indicates
                the order of the modes, e1 indicates the permittivity of
                the core, and e2 indicates the permittivity of the clad.
        Returns:
            vals: A 2D array of 4X4 floats indicating the Jacobian of the
                characteristic equations.
        """
        w, pol, n, e1, e2 = args
        ur, ui, vr, vi = uv
        u = ur + 1j * ui
        v = vr + 1j * vi
        sin = np.sin(u)
        cos = np.cos(u)
        if pol == "E":
            if n % 2 == 0:
                dv_du = (cos - u / sin) / sin
                dv_dv = 1.0
            else:
                dv_du = (sin + u / cos) / cos
                dv_dv = -1.0
        else:
            if n % 2 == 0:
                dv_du = (sin + u / cos) / cos
                dv_dv = -e1 / e2
            else:
                dv_du = (cos - u / sin) / sin
                dv_dv = e1 / e2
        vals = [
            [dv_du.real, dv_du.imag, dv_dv.real, dv_dv.imag],
            [dv_du.imag, dv_du.real, dv_dv.imag, dv_dv.real],
            [2 * u.real, 2 * u.imag, 2 * v.real, 2 * v.imag],
            [2 * u.imag, 2 * u.real, 2 * v.imag, 2 * v.real],
        ]
        return np.array(vals)

    def func_jac(self, uv, args):
        """Return the values and Jacobian of characteristic equations.

        Args:
            uv: A 1D array of 4 floats indicating (Re[u], Im[u], Re[v], Im[v]).
            args: A tuple (w, pol, n, e1, e2), where w indicates the angular
                frequency, pol indicates the polarization, n indicates
                the order of the modes, e1 indicates the permittivity of
                the core, and e2 indicates the permittivity of the clad.
        Returns:
            fs: A 1D array of 4 floats indicating the left-hand value of the
                characteristic equations.
            jas: A 2D array of 4X4 floats indicating the Jacobian of the
                characteristic equations.
        """
        w, pol, n, e1, e2 = args
        w2 = (self.r * w / 2) ** 2 * (e1 - e2)
        ur, ui, vr, vi = uv
        u = ur + 1j * ui
        v = vr + 1j * vi
        f2 = u ** 2 + v ** 2 - w2
        sin = np.sin(u)
        cos = np.cos(u)
        tan = np.tan(u)
        if pol == "E":
            if n % 2 == 0:
                f1 = u / tan + v
                dv_du = (cos - u / sin) / sin
                dv_dv = 1.0
            else:
                f1 = u * tan - v
                dv_du = (sin + u / cos) / cos
                dv_dv = -1.0
        else:
            if n % 2 == 0:
                f1 = u * tan - e1 * (v / e2)
                dv_du = (sin + u / cos) / cos
                dv_dv = -e1 / e2
            else:
                f1 = u / tan + e1 * (v / e2)
                dv_du = (cos - u / sin) / sin
                dv_dv = e1 / e2
        fs = np.array([f1.real, f1.imag, f2.real, f2.imag])
        jas = np.array(
            [
                [dv_du.real, dv_du.imag, dv_dv.real, dv_dv.imag],
                [dv_du.imag, dv_du.real, dv_dv.imag, dv_dv.real],
                [2 * u.real, 2 * u.imag, 2 * v.real, 2 * v.imag],
                [2 * u.imag, 2 * u.real, 2 * v.imag, 2 * v.real],
            ]
        )
        return fs, jas

    def beta2_w_min(self, pol, num_n):
        """Return roots and convergences of the characteristic equation at
            the lowest angular frequency, ws[0].

        Args:
            pol: 'E' or 'M' indicating the polarization.
            num_n: An integer indicating the number of modes.
        Returns:
            xs: A 1D array indicating the roots, whose length is 2.
            success: A 1D array indicating the convergence information for xs.
        """
        if self.clad(self.ws[0]).real < -1e7:
            xs = self.beta2_pec(self.ws[0], num_n)
            success = np.ones(num_n, dtype=bool)
            if pol == "E":
                success[0] = False
            return xs, success
        w_0 = 0.1
        e1 = self.fill(w_0)
        e2_0 = -5.0e5 + self.clad(w_0).imag * 1j
        de2 = (self.clad(w_0) - e2_0) / 1000
        xis = xs = self.beta2_pec(w_0, num_n)
        success = np.ones_like(xs, dtype=bool)
        for i in range(1001):
            e2 = e2_0 + de2 * i
            xs, success = self.beta2(w_0, pol, num_n, e1, e2, xis)
            for _, ok in enumerate(success):
                if not ok:
                    xs[_] = xis[_]
            xis = xs
        dw = (self.ws[0] - w_0) / 1000
        for i in range(1001):
            w = w_0 + dw * i
            e1 = self.fill(w)
            e2 = self.clad(w)
            xs, success = self.beta2(w, pol, num_n, e1, e2, xis)
            for _, ok in enumerate(success):
                if not ok:
                    xs[_] = xis[_]
            xis = xs
        return xs, success

    def beta2_w_max(self, pol, num_n):
        """Return roots and convergences of the characteristic equation at
            the highest angular frequency, ws[-1].

        Args:
            pol: 'E' or 'M' indicating the polarization.
            num_n: An integer indicating the number of modes.
        Returns:
            xs: A 1D array indicating the roots, whose length is 2.
            success: A 1D array indicating the convergence information for xs.
        """
        w_0 = self.ws[-1]
        if self.clad(w_0).real < -1e7:
            xs = self.beta2_pec(w_0, num_n)
            success = np.ones(num_n, dtype=bool)
            if pol == "E":
                success[0] = False
            return xs, success
        xs = self.beta2_pec(w_0, num_n)
        success = np.ones_like(xs, dtype=bool)
        e1 = self.fill(w_0)
        e2 = self.clad(w_0)
        e2_1 = -1.0e8 + self.clad(w_0).imag * 1j
        for j in range(10):
            e2_0 = e2_1
            de = (e2 - e2_0) / 100
            for i in range(100):
                e2_1 = e2_0 + de * i
                xs, success = self.beta2(w_0, pol, num_n, e1, e2_1, xs)
        e2_0 = e2_1
        de = (e2 - e2_0) / 100
        for i in range(101):
            e2_1 = e2_0 + de * i
            xs, success = self.beta2(w_0, pol, num_n, e1, e2_1, xs)
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
        betas = {}
        convs = {}
        for n in range(num_n):
            betas[("M", n, 1)] = np.zeros((len(self.ws), len(self.wis)), dtype=complex)
            convs[("M", n, 1)] = np.zeros((len(self.ws), len(self.wis)), dtype=bool)
            betas[("E", n, 1)] = np.zeros((len(self.ws), len(self.wis)), dtype=complex)
            convs[("E", n, 1)] = np.zeros((len(self.ws), len(self.wis)), dtype=bool)
        for iwr in range(num_iwr):
            for iwi in range(num_iwi):
                j = iwr * num_iwi + iwi
                w = self.ws[iwr] + 1j * self.wis[iwi]
                e2 = self.clad(w)
                for n in range(num_n):
                    x = xs_success_list[j][0][0][n]
                    v = self.v(x, w, e2)
                    betas[("M", n, 1)][iwr, iwi] = self.beta_from_beta2(x)
                    convs[("M", n, 1)][iwr, iwi] = (
                        xs_success_list[j][1][0][n] if v.real > abs(v.imag) else False
                    )
                    x = xs_success_list[j][0][1][n]
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

    def task(self, arg: tuple[str, int]) -> tuple[np.ndarray, np.ndarray]:
        """Return a dict of the roots of the characteristic equation

        Args:
            arg: (pol, num_n)
                pol: 'E' or 'M' indicating the polarization.
                num_n: The number of modes.
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
        pol, num_n = arg
        num_ws = len(self.ws)
        xs_array = np.zeros((num_ws, len(self.wis), num_n), dtype=complex)
        success_array = np.zeros((num_ws, len(self.wis), num_n), dtype=bool)
        iwr = iwi = 0
        wi = self.wis[iwi]
        xis, success = self.beta2_w_min(pol, num_n)
        xs_array[iwr, iwi] = xis
        success_array[iwr, iwi] = success
        for iwr in range(1, len(self.ws)):
            wr = self.ws[iwr]
            w = wr + 1j * wi
            e1 = self.fill(w)
            e2 = self.clad(w)
            xs, success = self.beta2(w, pol, num_n, e1, e2, xis)
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
                xs, success = self.beta2(w, pol, num_n, e1, e2, xis)
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
        return xs_array, success_array


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
        for i_pol, x0s in enumerate(xis_list):
            if i_pol == 0:
                pol = "M"
            else:
                pol = "E"
            xis = xs = x0s
            success = np.ones_like(xs, dtype=bool)
            for i in range(1, 16):
                self.clad.im_factor = 0.7 ** i
                if i == 15 or self.clad.im_factor < im_factor:
                    self.clad.im_factor = im_factor
                e2 = self.clad(w)
                xs, success = self.beta2(w, pol, num_n, e1, e2, xis)
                for _, ok in enumerate(success):
                    if not ok:
                        xs[_] = xis[_]
                xis = xs
            xs_list.append(xs)
            success_list.append(success)
        return xs_list, success_list
