from __future__ import annotations

import cmath
from typing import Optional

import numpy as np
import psutil
import ray
import scipy.special as ssp

from pymwm.utils import coax_utils
from pymwm.waveguide import Database, Sampling, Waveguide

from .samples import Samples, SamplesForRay, SamplesLowLoss, SamplesLowLossForRay


class Coax(Waveguide):
    """A class defining a coax waveguide."""

    def __init__(self, params: dict) -> None:
        """Init Coax class.

        Args:
            params: A dict whose keys and values are as follows:
                'core': A dict of the setting parameters of the core:
                    'shape': A string indicating the shape of the core.
                    'ri': A float indicating the inner radius [um].
                    'r': A float indicating the outer radius [um].
                    'fill': A dict of the parameters of the core Material.
                'clad': A dict of the parameters of the clad Material.
                'bounds': A dict indicating the bounds of database.interpolation
                    and its keys and values are as follows:
                    'wl_max': A float indicating the maximum wavelength [um]
                    'wl_min': A float indicating the minimum wavelength [um]
                    'wl_imag': A float indicating the maximum value of
                        abs(c / f_imag) [um] where f_imag is the imaginary part
                        of the frequency.
                'modes': A dict of the settings for calculating modes:
                    'wl_max': A float indicating the maximum wavelength [um]
                        (default: 5.0)
                    'wl_min': A float indicating the minimum wavelength [um]
                        (default: 0.4)
                    'wl_imag': A float indicating the maximum value of
                        abs(c / f_imag) [um] where f_imag is the imaginary part
                        of the frequency. (default: 5.0)
                    'dw': A float indicating frequency interval
                        [rad c / 1um]=[2.99792458e14 rad / s]
                        (default: 1 / 64).
                    'num_n': An integer indicating the number of orders of
                        modes.
                    'num_m': An integer indicating the number of modes in each
                        order and polarization.
                    'ls': A list of characters chosen from "h" (horizontal
                        polarization) and "v" (vertical polarization).
        """
        self.ri = params["core"]["ri"]
        params["core"]["size"] = params["core"]["r"]
        params["core"]["size2"] = self.ri
        super().__init__(params)
        if self.clad.label == "PEC":
            from pymwm.cutoff import Cutoff

            co = Cutoff(self.num_n, self.num_m)
            self.co_list = []
            for n in range(self.num_n):
                co_per_n = []
                for pol, m_end in [("M", self.num_m + 2), ("E", self.num_m + 1)]:
                    for m in range(1, m_end):
                        alpha = (pol, n, m)
                        co_per_n.append(co(alpha, self.ri / self.r))
                self.co_list.append(np.array(co_per_n))
        else:
            self.co_list = self.samples.co_list

    def get_alphas(self, alpha_list: list[tuple[str, int, int]]) -> dict:
        alphas: dict = {"h": [], "v": []}
        for alpha in [("E", 0, m) for m in range(1, self.num_m + 1)]:
            if alpha in alpha_list:
                alphas["v"].append(alpha)
        for alpha in [
            ("E", n, m) for n in range(1, self.num_n) for m in range(1, self.num_m + 1)
        ]:
            if alpha in alpha_list:
                alphas["h"].append(alpha)
                alphas["v"].append(alpha)
        for alpha in [("M", 0, m) for m in range(1, self.num_m + 1)]:
            if alpha in alpha_list:
                alphas["h"].append(alpha)
        for alpha in [
            ("M", n, m) for n in range(1, self.num_n) for m in range(1, self.num_m + 1)
        ]:
            if alpha in alpha_list:
                alphas["h"].append(alpha)
                alphas["v"].append(alpha)
        return alphas

    def betas_convs_samples(self, params: dict) -> tuple[dict, dict, Samples]:
        im_factor = self.clad.im_factor
        self.clad.im_factor = 1.0
        self.clad_params["im_factor"] = 1.0
        p_modes = params["modes"].copy()
        num_n_0 = p_modes["num_n"]
        num_m_0 = p_modes["num_m"]
        betas: dict = {}
        convs: dict = {}
        success = False
        catalog = Database().load_catalog()
        num_n_max = catalog["num_n"].max()
        num_m_max = catalog["num_m"].max()
        if not np.isnan(num_n_max):
            for num_n, num_m in [
                (n, m)
                for n in range(num_n_0, num_n_max + 1)
                for m in range(num_m_0, num_m_max + 1)
            ]:
                p_modes["num_n"] = num_n
                p_modes["num_m"] = num_m
                smp = Samples(
                    self.r, self.fill_params, self.clad_params, p_modes, self.ri
                )
                try:
                    betas, convs = smp.database.load()
                    success = True
                    break
                except IndexError:
                    continue
        if not success:
            p_modes["num_n"] = num_n_0
            p_modes["num_m"] = num_m_0
            betas, convs, smp = self.do_sampling(p_modes)
        if im_factor != 1.0:
            self.clad.im_factor = im_factor
            self.clad_params["im_factor"] = im_factor
            betas, convs, smp = self.do_sampling_for_im_factor(betas, convs, p_modes)
        return betas, convs, smp

    def do_sampling(self, p_modes: dict) -> tuple[dict, dict, Samples]:
        num_n_0 = p_modes["num_n"]
        num_m_0 = p_modes["num_m"]
        smp = Samples(self.r, self.fill_params, self.clad_params, p_modes, self.ri)
        ray.shutdown()
        try:
            ray.init()
            p_modes_id = ray.put(p_modes)
            pool = ray.util.ActorPool(
                SamplesForRay.remote(
                    self.r, self.fill_params, self.clad_params, p_modes_id, self.ri
                )
                for _ in range(psutil.cpu_count())
            )
            xs_success_wr_list: list[tuple[np.ndarray, np.ndarray]] = list(
                pool.map(lambda a, arg: a.wr_sampling.remote(arg), range(num_n_0))
            )
            num_wr = xs_success_wr_list[0][0].shape[0]
            args = []
            for n in range(num_n_0):
                xs_array, _ = xs_success_wr_list[n]
                for iwr in range(num_wr):
                    args.append((n, iwr, xs_array[iwr]))
            xs_success_wi_list: list[tuple[np.ndarray, np.ndarray]] = list(
                pool.map(lambda a, arg: a.wi_sampling.remote(arg), args)
            )
            num_wi = xs_success_wi_list[0][0].shape[0]
            xs_success_list: list[tuple[np.ndarray, np.ndarray]] = []
            for n in range(num_n_0):
                xs_array = np.zeros((num_wr, num_wi, 2 * num_m_0 + 1), dtype=complex)
                success_array = np.zeros((num_wr, num_wi, 2 * num_m_0 + 1), dtype=bool)
                for iwr in range(num_wr):
                    i = num_wr * n + iwr
                    xs_i, success_i = xs_success_wi_list[i]
                    xs_array[iwr] = xs_i
                    success_array[iwr] = success_i
                xs_success_list.append((xs_array, success_array))
        finally:
            ray.shutdown()
        betas, convs = smp.betas_convs(xs_success_list)
        smp.database.save(betas, convs)
        return betas, convs, smp

    def do_sampling_for_im_factor(
        self, betas: dict, convs: dict, p_modes: dict
    ) -> tuple[dict, dict, SamplesLowLoss]:
        smp = SamplesLowLoss(
            self.r, self.fill_params, self.clad_params, p_modes, self.ri
        )
        try:
            betas, convs = smp.database.load()
        except IndexError:
            num_n = p_modes["num_n"]
            num_m = p_modes["num_m"]
            args = []
            for iwr in range(len(smp.ws)):
                for iwi in range(len(smp.wis)):
                    xis_list = []
                    for n in range(num_n):
                        xis = []
                        for i in range(num_m + 1):
                            xis.append(betas[("M", n, i + 1)][iwr, iwi] ** 2)
                        for i in range(num_m):
                            xis.append(betas[("E", n, i + 1)][iwr, iwi] ** 2)
                        xis_list.append(xis)
                    args.append((iwr, iwi, xis_list))
            try:
                ray.init()
                p_modes_id = ray.put(p_modes)
                pool = ray.util.ActorPool(
                    SamplesLowLossForRay.remote(
                        self.r, self.fill_params, self.clad_params, p_modes_id, self.ri
                    )
                    for _ in range(psutil.cpu_count())
                )
                xs_success_list = list(
                    pool.map(lambda a, arg: a.task.remote(arg), args)
                )
            finally:
                ray.shutdown()
            betas, convs = smp.betas_convs(xs_success_list)
            smp.database.save(betas, convs)
        return betas, convs, smp

    def beta(self, w: complex, alpha: tuple[str, int, int]) -> complex:
        """Return phase constant

        Args:
            w: A complex indicating the angular frequency
            alpha: (pol, n, m)
                pol: 'M' (TM-like mode) or 'E' (TE-like mode)
                n: The order of the mode
                m: The sub order of the mode.
        Returns:
            h: The phase constant.
        """
        if self.clad.label == "PEC":
            return self.beta_pec(w, alpha)
        wr = w.real
        wi = w.imag
        hr: float = self.beta_funcs[(alpha, "real")](wr, wi)[0, 0]
        hi: float = self.beta_funcs[(alpha, "imag")](wr, wi)[0, 0]
        # if hr < 0:
        #     hr = 1e-16
        # if hi < 0:
        #     hi = 1e-16
        return hr + 1j * hi

    def beta_pec(self, w: complex, alpha: tuple[str, int, int]) -> complex:
        """Return phase constant of PEC waveguide

        Args:
            w: A complex indicating the angular frequency
            alpha: A tuple (pol, n, m) where pol is 'M' for TM mode or
                'E' for TE mode, n is the order of the mode, and m is the
                number of modes in the order and the polarization.
        Returns:
            h: A complex indicating the phase constant.
        """
        w_comp = w.real + 1j * w.imag
        pol, n, m = alpha
        if pol == "M":
            chi = self.co_list[n][m - 1]
        else:
            chi = self.co_list[n][self.num_m + m]
        val = cmath.sqrt(self.fill(w_comp) * w_comp ** 2 - chi ** 2 / self.r ** 2)
        if abs(val.real) > abs(val.imag):
            if val.real < 0:
                val *= -1
        else:
            if val.imag < 0:
                val *= -1
        return val

    def coef(self, h: complex, w: complex, alpha: tuple[str, int, int]) -> tuple:
        """Return the coefficients of TE- and TM- components which compose
        the hybrid mode.

        Args:
            h: Phase constant.
            w: Angular frequency
            alpha: A tuple (pol, n, m) where pol is 'M' for TM-like mode or
                'E' for TE-like mode, n is the order of the mode, and m is
                the number of modes in the order and the polarization.
        Returns:
            array([a1, b1, a2, b2, c2, d2, a3, b3]):
                a1: Coefficient of TE-component for core rod
                b1: Coefficient of TM-component for core rod
                a2: Coefficient of TE-component described by Jn for dielectric region
                b2: Coefficient of TM-component described by Jn for dielectric region
                c2: Coefficient of TE-component described by Yn for dielectric region
                d2: Coefficient of TM-component described by Yn for dielectric region
                a3: Coefficient of TE-component for clad metal
                b3: Coefficient of TM-component for clad metal
        """
        pol, n, m = alpha
        w = w.real + 1j * w.imag
        h = h.real + 1j * h.imag
        e1 = self.fill(w)
        e2 = self.clad(w)
        ee = e1 / e2
        u = self.samples.u(h ** 2, w, e1)
        ju = ssp.jv(n, u)
        jpu = -ssp.jv(n + 1, u) + n / u * ju

        yu = ssp.yv(n, u)
        ypu = -ssp.yv(n + 1, u) + n / u * yu
        if e2.real < -1e6:
            a1 = b1 = a3 = b3 = 0.0j
            if pol == "TE":
                a2 = 1.0 + 0.0j
                c2 = -jpu / ypu
                b2 = d2 = 0.0j
            else:
                b2 = 1.0 + 0.0j
                d2 = -ju / yu
                a2 = c2 = 0.0j
        else:
            hew = h ** 2 / e2 / w ** 2
            x = self.samples.x(h ** 2, w, e1)
            y = self.samples.y(h ** 2, w, e2)
            v = self.samples.v(h ** 2, w, e2)

            kv = ssp.kv(n, v)
            kpv = -ssp.kv(n + 1, v) + n / v * kv

            jx = ssp.jv(n, x)
            jpx = -ssp.jv(n + 1, x) + n / x * jx

            yx = ssp.yv(n, x)
            ypx = -ssp.yv(n + 1, x) + n / x * yx

            iy = ssp.iv(n, y)
            ipy = ssp.iv(n + 1, y) + n / y * iy

            nuv = n * (v / u + u / v)
            nxy = n * (y / x + x / y)

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

            if pol == "E":
                a2 = 1.0 + 0j
                A = a[1:, 1:]
                B = -a[1:, 0]
                c2, b2, d2 = np.linalg.solve(A, B)
            else:
                b2 = 1.0 + 0j
                A = a[[0, 1, 3]][:, [0, 1, 3]]
                B = -a[[0, 1, 3]][:, 2]
                a2, c2, d2 = np.linalg.solve(A, B)
            a1 = -x / (y * iy) * (jx * a2 + yx * c2)
            b1 = -x / (y * iy) * (jx * b2 + yx * d2)
            a3 = -u / (v * kv) * (ju * a2 + yu * c2)
            b3 = -u / (v * kv) * (ju * b2 + yu * d2)
        vals = (a1, b1, a2, b2, c2, d2, a3, b3)
        norm = self.norm(h, w, alpha, vals)
        return tuple(val / norm for val in vals)

    def norm(
        self, h: complex, w: complex, alpha: tuple[str, int, int], coef: tuple
    ) -> complex:
        a1, b1, a2, b2, c2, d2, a3, b3 = coef
        r, ri = self.r, self.ri
        pol, n, m = alpha
        w = w.real + 1j * w.imag
        h = h.real + 1j * h.imag
        e1 = self.fill(w)
        e2 = self.clad(w)
        en = 1 if n == 0 else 2
        u = self.samples.u(h ** 2, w, e1)
        x = self.samples.x(h ** 2, w, e1)
        ju = ssp.jv(n, u)
        jpu = -ssp.jv(n + 1, u) + n / u * ju
        yu = ssp.yv(n, u)
        ypu = -ssp.yv(n + 1, u) + n / u * yu
        jx = ssp.jv(n, x)
        jpx = -ssp.jv(n + 1, x) + n / x * jx
        yx = ssp.yv(n, x)
        ypx = -ssp.yv(n + 1, x) + n / x * yx
        I2 = cmath.pi * (
            1
            / en
            * (
                (
                    r ** 2
                    * (jpu ** 2 + (1 - n ** 2 / u ** 2) * ju ** 2 + 2 * jpu * ju / u)
                    - ri ** 2
                    * (jpx ** 2 + (1 - n ** 2 / x ** 2) * jx ** 2 + 2 * jpx * jx / x)
                )
                * (a2 ** 2 + b2 ** 2)
                + (
                    r ** 2
                    * (ypu ** 2 + (1 - n ** 2 / u ** 2) * yu ** 2 + 2 * ypu * yu / u)
                    - ri ** 2
                    * (ypx ** 2 + (1 - n ** 2 / x ** 2) * yx ** 2 + 2 * ypx * yx / x)
                )
                * (c2 ** 2 + d2 ** 2)
                + 2
                * (
                    r ** 2
                    * (jpu * ypu + (1 - n ** 2 / u ** 2) * ju * yu + 2 * jpu * yu / u)
                    - ri ** 2
                    * (jpx * ypx + (1 - n ** 2 / x ** 2) * jx * yx + 2 * jpx * yx / x)
                )
                * (a2 * c2 + b2 * d2)
            )
            + 2
            * n
            * (
                (r ** 2 / u ** 2 * ju ** 2 - ri ** 2 / x ** 2 * jx ** 2) * a2 * b2
                + (r ** 2 / u ** 2 * yu ** 2 - ri ** 2 / x ** 2 * yx ** 2) * c2 * d2
                + (r ** 2 / u ** 2 * ju * yu - ri ** 2 / x ** 2 * jx * yx)
                * (a2 * d2 + b2 * c2)
            )
        )
        if e2.real < -1e6:
            return cmath.sqrt(I2)
        else:
            v = self.samples.v(h ** 2, w, e2)
            y = self.samples.y(h ** 2, w, e2)
            kv = ssp.kv(n, v)
            kpv = -ssp.kv(n + 1, v) + n / v * kv
            iy = ssp.iv(n, y)
            ipy = ssp.iv(n + 1, y) + n / y * iy
            I1 = (
                cmath.pi
                * ri ** 2
                * (
                    1
                    / en
                    * (ipy ** 2 - (1 + n ** 2 / y ** 2) * iy ** 2 + 2 * ipy * iy / y)
                    * (a1 ** 2 + b1 ** 2)
                    + 2 * n * iy ** 2 / y ** 2 * a1 * b1
                )
            )
            I3 = (
                -cmath.pi
                * r ** 2
                * (
                    1
                    / en
                    * (kpv ** 2 - (1 + n ** 2 / v ** 2) * kv ** 2 + 2 * kpv * kv / v)
                    * (a3 ** 2 + b3 ** 2)
                    + 2 * n * kv ** 2 / v ** 2 * a3 * b3
                )
            )
            return cmath.sqrt(I1 + I2 + I3)

    @staticmethod
    def y_te(w, h):
        return h / w

    def y_tm_core(self, w, h):
        e = self.fill(w)
        return e * w / h

    def y_tm_clad(self, w, h):
        e = self.clad(w)
        return e * w / h

    def Y(
        self, w: complex, h: complex, alpha: tuple[str, int, int], coef: tuple
    ) -> complex:
        """Return the effective admittance of the waveguide mode

        Args:
            w: Angular frequency
            h: Phase constant.
            alpha: A tuple (pol, n, m) where pol is 'M' for TM-like mode or
                'E' for TE-like mode, n is the order of the mode, and m is
                the number of modes in the order and the polarization.
            coef: array([a1, b1, a2, b2, c2, d2, a3, b3])
                a1: Coefficient of TE-component for core rod
                b1: Coefficient of TM-component for core rod
                a2: Coefficient of TE-component described by Jn for dielectric region
                b2: Coefficient of TM-component described by Jn for dielectric region
                c2: Coefficient of TE-component described by Yn for dielectric region
                d2: Coefficient of TM-component described by Yn for dielectric region
                a3: Coefficient of TE-component for clad metal
                b3: Coefficient of TM-component for clad metal
        Returns:
            y: Effective admittance
        """
        a1, b1, a2, b2, c2, d2, a3, b3 = coef
        r, ri = self.r, self.ri
        pol, n, m = alpha
        w = w.real + 1j * w.imag
        h = h.real + 1j * h.imag
        e1 = self.fill(w)
        e2 = self.clad(w)
        en = 1 if n == 0 else 2
        y_te = Coax.y_te(w, h)
        y_tm1 = self.y_tm_core(w, h)
        y_tm2 = self.y_tm_clad(w, h)
        u = self.samples.u(h ** 2, w, e1)
        x = self.samples.x(h ** 2, w, e1)
        ju = ssp.jv(n, u)
        jpu = -ssp.jv(n + 1, u) + n / u * ju
        yu = ssp.yv(n, u)
        ypu = -ssp.yv(n + 1, u) + n / u * yu
        jx = ssp.jv(n, x)
        jpx = -ssp.jv(n + 1, x) + n / x * jx
        yx = ssp.yv(n, x)
        ypx = -ssp.yv(n + 1, x) + n / x * yx
        I2 = cmath.pi * (
            1
            / en
            * (
                (
                    r ** 2
                    * (jpu ** 2 + (1 - n ** 2 / u ** 2) * ju ** 2 + 2 * jpu * ju / u)
                    - ri ** 2
                    * (jpx ** 2 + (1 - n ** 2 / x ** 2) * jx ** 2 + 2 * jpx * jx / x)
                )
                * (y_te * a2 ** 2 + y_tm1 * b2 ** 2)
                + (
                    r ** 2
                    * (ypu ** 2 + (1 - n ** 2 / u ** 2) * yu ** 2 + 2 * ypu * yu / u)
                    - ri ** 2
                    * (ypx ** 2 + (1 - n ** 2 / x ** 2) * yx ** 2 + 2 * ypx * yx / x)
                )
                * (y_te * c2 ** 2 + y_tm1 * d2 ** 2)
                + 2
                * (
                    r ** 2
                    * (jpu * ypu + (1 - n ** 2 / u ** 2) * ju * yu + 2 * jpu * yu / u)
                    - ri ** 2
                    * (jpx * ypx + (1 - n ** 2 / x ** 2) * jx * yx + 2 * jpx * yx / x)
                )
                * (y_te * a2 * c2 + y_tm1 * b2 * d2)
            )
            + n
            * (y_te + y_tm1)
            * (
                (r ** 2 / u ** 2 * ju ** 2 - ri ** 2 / x ** 2 * jx ** 2) * a2 * b2
                + (r ** 2 / u ** 2 * yu ** 2 - ri ** 2 / x ** 2 * yx ** 2) * c2 * d2
                + (r ** 2 / u ** 2 * ju * yu - ri ** 2 / x ** 2 * jx * yx)
                * (a2 * d2 + b2 * c2)
            )
        )
        if e2.real < -1e6:
            return I2
        else:
            v = self.samples.v(h ** 2, w, e2)
            y = self.samples.y(h ** 2, w, e2)
            kv = ssp.kv(n, v)
            kpv = -ssp.kv(n + 1, v) + n / v * kv
            iy = ssp.iv(n, y)
            ipy = ssp.iv(n + 1, y) + n / y * iy

            I1 = (
                cmath.pi
                * ri ** 2
                * (
                    1
                    / en
                    * (ipy ** 2 - (1 + n ** 2 / y ** 2) * iy ** 2 + 2 * ipy * iy / y)
                    * (y_te * a1 ** 2 + y_tm2 * b1 ** 2)
                    + n * iy ** 2 / y ** 2 * (y_te + y_tm2) * a1 * b1
                )
            )
            I3 = (
                -cmath.pi
                * r ** 2
                * (
                    1
                    / en
                    * (kpv ** 2 - (1 + n ** 2 / v ** 2) * kv ** 2 + 2 * kpv * kv / v)
                    * (y_te * a3 ** 2 + y_tm2 * b3 ** 2)
                    + n * kv ** 2 / v ** 2 * (y_te + y_tm2) * a3 * b3
                )
            )
            return I1 + I2 + I3

    def coefs(self, hs, w):
        A1s = []
        B1s = []
        A2s = []
        B2s = []
        C2s = []
        D2s = []
        A3s = []
        B3s = []
        for h, s, n, m in zip(hs, self.s_all, self.n_all, self.m_all):
            pol = "E" if s == 0 else "M"
            a1, b1, a2, b2, c2, d2, a3, b3 = self.coef(h, w, (pol, n, m))
            A1s.append(a1)
            B1s.append(b1)
            A2s.append(a2)
            B2s.append(b2)
            C2s.append(c2)
            D2s.append(d2)
            A3s.append(a3)
            B3s.append(b3)
        return (
            np.ascontiguousarray(A1s),
            np.ascontiguousarray(B1s),
            np.ascontiguousarray(A2s),
            np.ascontiguousarray(B2s),
            np.ascontiguousarray(C2s),
            np.ascontiguousarray(D2s),
            np.ascontiguousarray(A3s),
            np.ascontiguousarray(B3s),
        )

    def Ys(self, w, hs, A1s, B1s, A2s, B2s, C2s, D2s, A3s, B3s):
        vals = []
        coefs = zip(A1s, B1s, A2s, B2s, C2s, D2s, A3s, B3s)
        for h, s, n, coef in zip(hs, self.s_all, self.n_all, coefs):
            pol = "E" if s == 0 else "M"
            vals.append(self.Y(w, h, (pol, n, 1), coef))
        return np.array(vals)

    def props_numpy(self, w):
        e1 = self.fill(w)
        e2 = self.clad(w)
        hs = np.array([self.beta(w, alpha) for alpha in self.alpha_all])
        A1s, B1s, A2s, B2s, C2s, D2s, A3s, B3s = self.coefs(hs, w)
        Ys = self.Ys(w, hs, A1s, B1s, A2s, B2s, C2s, D2s, A3s, B3s)
        xs = self.samples.x(hs ** 2, w, e1)
        ys = self.samples.y(hs ** 2, w, e2)
        us = self.samples.u(hs ** 2, w, e1)
        vs = self.samples.v(hs ** 2, w, e2)
        jxs = ssp.jv(self.n_all, xs)
        jpxs = ssp.jvp(self.n_all, xs)
        yxs = ssp.yv(self.n_all, xs)
        ypxs = ssp.yvp(self.n_all, xs)
        jus = ssp.jv(self.n_all, us)
        jpus = ssp.jvp(self.n_all, us)
        yus = ssp.yv(self.n_all, us)
        ypus = ssp.yvp(self.n_all, us)
        if e2.real < -1e6:
            iys = np.inf * np.ones_like(ys)
            ipys = np.inf * np.ones_like(ys)
            kvs = np.zeros_like(vs)
            kpvs = np.zeros_like(vs)
        else:
            iys = ssp.iv(self.n_all, ys)
            ipys = ssp.ivp(self.n_all, ys)
            kvs = ssp.kv(self.n_all, vs)
            kpvs = ssp.kvp(self.n_all, vs)
        return (
            hs,
            xs,
            ys,
            us,
            vs,
            jxs,
            jpxs,
            yxs,
            ypxs,
            iys,
            ipys,
            jus,
            jpus,
            yus,
            ypus,
            kvs,
            kpvs,
            A1s,
            B1s,
            A2s,
            B2s,
            C2s,
            D2s,
            A3s,
            B3s,
            Ys,
        )

    def props(self, w):
        e1 = self.fill(w)
        e2 = self.clad(w)
        hs = np.array([self.beta(w, alpha) for alpha in self.alpha_all])
        (
            xs,
            ys,
            us,
            vs,
            jxs,
            jpxs,
            yxs,
            ypxs,
            iys,
            ipys,
            jus,
            jpus,
            yus,
            ypus,
            kvs,
            kpvs,
            A1s,
            B1s,
            A2s,
            B2s,
            C2s,
            D2s,
            A3s,
            B3s,
            Ys,
        ) = coax_utils.props_cython(
            w, self.r, self.ri, self.s_all, self.n_all, hs, e1, e2
        )
        return (
            hs,
            xs,
            ys,
            us,
            vs,
            jxs,
            jpxs,
            yxs,
            ypxs,
            iys,
            ipys,
            jus,
            jpus,
            yus,
            ypus,
            kvs,
            kpvs,
            A1s,
            B1s,
            A2s,
            B2s,
            C2s,
            D2s,
            A3s,
            B3s,
            Ys,
        )

    def e_field_r_dep(
        self,
        r: float,
        w: complex,
        alpha: tuple[str, int, int],
        h: complex,
        coef: tuple,
    ) -> np.ndarray:
        """Return the r-dependence of the electric field vectors in cylindrical coordinate

        Args:
            r: The distance from origin [um].
            w: The angular frequency.
            alpha: (pol, n, m)
                pol: 'M' (TM-like mode) or 'E' (TE-like mode).
                n: The order of the mode.
                m: The sub order of the mode.
            h: The complex phase constant.
            coef: The coefficients of TE- and TM- components
        Returns:
            e_vec: An array of complexes [er, ep, ez].
        """
        a1, b1, a2, b2, c2, d2, a3, b3 = coef
        pol, n, m = alpha
        w = w.real + 1j * w.imag
        h = h.real + 1j * h.imag
        e1 = self.fill(w)
        e2 = self.clad(w)
        if r < self.ri:
            _y = self.samples.y(h ** 2, w, e2)
            yr = _y * r / self.ri
            iy = ssp.iv(n, yr)
            iy_plus = ssp.iv(n + 1, yr)
            iy_minus = ssp.iv(n - 1, yr)
            niy_y = (iy_minus - iy_plus) / 2
            ipy = (iy_minus + iy_plus) / 2
            er = niy_y * a1 + ipy * b1
            ep = ipy * a1 + niy_y * b1
            ez = -_y / (1j * h * self.ri) * iy * b1
        elif self.ri <= r < self.r:
            u = self.samples.u(h ** 2, w, e1)
            ur = u * r / self.r
            ju = ssp.jv(n, ur)
            ju_plus = ssp.jv(n + 1, ur)
            ju_minus = ssp.jv(n - 1, ur)
            nju_u = (ju_minus + ju_plus) / 2
            jpu = (ju_minus - ju_plus) / 2
            yu = ssp.yv(n, ur)
            yu_plus = ssp.yv(n + 1, ur)
            yu_minus = ssp.yv(n - 1, ur)
            nyu_u = (yu_minus + yu_plus) / 2
            ypu = (yu_minus - yu_plus) / 2
            er = nju_u * a2 + jpu * b2 + nyu_u * c2 + ypu * d2
            ep = jpu * a2 + nju_u * b2 + ypu * c2 + nyu_u * d2
            ez = u / (1j * h * self.r) * (ju * b2 + yu * d2)
        else:
            v = self.samples.v(h ** 2, w, e2)
            vr = v * r / self.r
            kv = ssp.kv(n, vr)
            kv_plus = ssp.kv(n + 1, vr)
            kv_minus = ssp.kv(n - 1, vr)
            nkv_v = -(kv_minus - kv_plus) / 2
            kpv = -(kv_minus + kv_plus) / 2
            er = nkv_v * a3 + kpv * b3
            ep = kpv * a3 + nkv_v * b3
            ez = -v / (1j * h * self.r) * kv * b3
        return np.array([er, ep, ez])

    def e_field(
        self,
        x: float,
        y: float,
        w: complex,
        dir: str,
        alpha: tuple[str, int, int],
        h: complex,
        coef: tuple,
    ) -> np.ndarray:
        """Return the electric field vectors for the specified mode and point

        Args:
            x: The x coordinate [um].
            y: The y coordinate [um].
            dir: "h" (horizontal polarization) or "v" (vertical polarization)
            w: The angular frequency.
            alpha: (pol, n, m)
                pol: 'M' (TM-like mode) or 'E' (TE-like mode).
                n: The order of the mode.
                m: The sub order of the mode.
            h: The complex phase constant.
            coef: The coefficients of TE- and TM- components
        Returns:
            e_vec: An array of complexes [ex, ey, ez].
        """
        pol, n, m = alpha
        r = np.hypot(x, y)
        p = np.arctan2(y, x)
        w = w.real + 1j * w.imag
        h = h.real + 1j * h.imag
        if dir == "h":
            fr = np.cos(n * p)
            fp = -np.sin(n * p)
        else:
            fr = np.sin(n * p)
            fp = np.cos(n * p)
        er, ep, ez = self.e_field_r_dep(r, w, alpha, h, coef)
        er *= fr
        ep *= fp
        ez *= fr
        ex = er * np.cos(p) - ep * np.sin(p)
        ey = er * np.sin(p) + ep * np.cos(p)
        return np.array([ex, ey, ez])

    def h_field_r_dep(
        self,
        r: float,
        w: complex,
        alpha: tuple[str, int, int],
        h: complex,
        coef: tuple,
    ) -> np.ndarray:
        """Return the r-dependence of the magnetic field vectors in cylindrical coordinate

        Args:
            r: The distance from origin [um].
            w: The angular frequency.
            alpha: (pol, n, m)
                pol: 'M' (TM-like mode) or 'E' (TE-like mode).
                n: The order of the mode.
                m: The sub order of the mode.
            h: The complex phase constant.
            coef: The coefficients of TE- and TM- components
        Returns:
            e_vec: An array of complexes [er, ep, ez].
        """
        a1, b1, a2, b2, c2, d2, a3, b3 = coef
        pol, n, m = alpha
        w = w.real + 1j * w.imag
        h = h.real + 1j * h.imag
        e1 = self.fill(w)
        e2 = self.clad(w)
        y_te = Coax.y_te(w, h)
        if r < self.ri:
            y_tm = self.y_tm_clad(w, h)
            _y = self.samples.y(h ** 2, w, e2)
            yr = _y * r / self.ri
            iy = ssp.iv(n, yr)
            iy_plus = ssp.iv(n + 1, yr)
            iy_minus = ssp.iv(n - 1, yr)
            niy_y = (iy_minus - iy_plus) / 2
            ipy = (iy_minus + iy_plus) / 2
            hr = -(y_te * ipy * a1 + y_tm * niy_y * b1)
            hp = y_te * niy_y * a1 + y_tm * ipy * b1
            hz = -1j * _y / self.ri * iy * a1
        elif self.ri <= r < self.r:
            y_tm = self.y_tm_core(w, h)
            u = self.samples.u(h ** 2, w, e1)
            ur = u * r / self.r
            ju = ssp.jv(n, ur)
            ju_plus = ssp.jv(n + 1, ur)
            ju_minus = ssp.jv(n - 1, ur)
            nju_u = (ju_minus + ju_plus) / 2
            jpu = (ju_minus - ju_plus) / 2
            yu = ssp.yv(n, ur)
            yu_plus = ssp.yv(n + 1, ur)
            yu_minus = ssp.yv(n - 1, ur)
            nyu_u = (yu_minus + yu_plus) / 2
            ypu = (yu_minus - yu_plus) / 2
            hr = -(
                y_te * jpu * a2
                + y_tm * nju_u * b2
                + y_te * ypu * c2
                + y_tm * nyu_u * d2
            )
            hp = (
                y_te * nju_u * a2
                + y_tm * jpu * b2
                + y_te * nyu_u * c2
                + y_tm * ypu * d2
            )
            hz = 1j * u / self.r * (ju * a2 + yu * c2)
        else:
            y_tm = self.y_tm_clad(w, h)
            v = self.samples.v(h ** 2, w, e2)
            vr = v * r / self.r
            kv = ssp.kv(n, vr)
            kv_plus = ssp.kv(n + 1, vr)
            kv_minus = ssp.kv(n - 1, vr)
            nkv_v = -(kv_minus - kv_plus) / 2
            kpv = -(kv_minus + kv_plus) / 2
            hr = -(y_te * kpv * a3 + y_tm * nkv_v * b3)
            hp = y_te * nkv_v * a3 + y_tm * kpv * b3
            hz = -1j * v / self.r * kv * a3
        return np.array([hr, hp, hz])

    def h_field(self, x, y, w, dir, alpha, h, coef):
        """Return the magnetic field vectors for the specified mode and
        point

        Args:
            x: A float indicating the x coordinate [um]
            y: A float indicating the y coordinate [um]
            w: A complex indicating the angular frequency
            dir: "h" (horizontal polarization) or "v" (vertical polarization)
            alpha: A tuple (pol, n, m) where pol is 'M' for TM-like mode or
                'E' for TE-like mode, n is the order of the mode, and m is
                the number of modes in the order and the polarization.
            h: A complex indicating the phase constant.
            coef: The coefficients of TE- and TM- components
        Returns:
            h_vec: An array of complexes [hx, hy, hz].
        """
        pol, n, m = alpha
        r = np.hypot(x, y)
        p = np.arctan2(y, x)
        w = w.real + 1j * w.imag
        h = h.real + 1j * h.imag
        if dir == "h":
            fr = np.cos(n * p)
            fp = -np.sin(n * p)
        else:
            fr = np.sin(n * p)
            fp = np.cos(n * p)
        hr, hp, hz = self.h_field_r_dep(r, w, alpha, h, coef)
        hr *= fp
        hp *= fr
        hz *= fp
        hx = hr * np.cos(p) - hp * np.sin(p)
        hy = hr * np.sin(p) + hp * np.cos(p)
        return np.array([hx, hy, hz])

    def field_r_dep(
        self,
        r: float,
        w: complex,
        alpha: tuple[str, int, int],
        h: complex,
        coef: tuple,
    ) -> np.ndarray:
        """Return the r-dependence of the field vectors in cylindrical coordinate

        Args:
            r: The distance from origin [um].
            w: The angular frequency.
            alpha: (pol, n, m)
                pol: 'M' (TM-like mode) or 'E' (TE-like mode).
                n: The order of the mode.
                m: The sub order of the mode.
            h: The complex phase constant.
            coef: The coefficients of TE- and TM- components
        Returns:
            e_vec: An array of complexes [er, ep, ez].
        """
        a1, b1, a2, b2, c2, d2, a3, b3 = coef
        pol, n, m = alpha
        w = w.real + 1j * w.imag
        h = h.real + 1j * h.imag
        e1 = self.fill(w)
        e2 = self.clad(w)
        y_te = Coax.y_te(w, h)
        if r < self.ri:
            y_tm = self.y_tm_clad(w, h)
            _y = self.samples.y(h ** 2, w, e2)
            yr = _y * r / self.ri
            iy = ssp.iv(n, yr)
            iy_plus = ssp.iv(n + 1, yr)
            iy_minus = ssp.iv(n - 1, yr)
            niy_y = (iy_minus - iy_plus) / 2
            ipy = (iy_minus + iy_plus) / 2
            er = niy_y * a1 + ipy * b1
            ep = ipy * a1 + niy_y * b1
            ez = -_y / (1j * h * self.ri) * iy * b1
            hr = -(y_te * ipy * a1 + y_tm * niy_y * b1)
            hp = y_te * niy_y * a1 + y_tm * ipy * b1
            hz = -1j * _y / self.ri * iy * a1
        elif self.ri <= r < self.r:
            y_tm = self.y_tm_core(w, h)
            u = self.samples.u(h ** 2, w, e1)
            ur = u * r / self.r
            ju = ssp.jv(n, ur)
            ju_plus = ssp.jv(n + 1, ur)
            ju_minus = ssp.jv(n - 1, ur)
            nju_u = (ju_minus + ju_plus) / 2
            jpu = (ju_minus - ju_plus) / 2
            yu = ssp.yv(n, ur)
            yu_plus = ssp.yv(n + 1, ur)
            yu_minus = ssp.yv(n - 1, ur)
            nyu_u = (yu_minus + yu_plus) / 2
            ypu = (yu_minus - yu_plus) / 2
            er = nju_u * a2 + jpu * b2 + nyu_u * c2 + ypu * d2
            ep = jpu * a2 + nju_u * b2 + ypu * c2 + nyu_u * d2
            ez = u / (1j * h * self.r) * (ju * b2 + yu * d2)
            hr = -(
                y_te * jpu * a2
                + y_tm * nju_u * b2
                + y_te * ypu * c2
                + y_tm * nyu_u * d2
            )
            hp = (
                y_te * nju_u * a2
                + y_tm * jpu * b2
                + y_te * nyu_u * c2
                + y_tm * ypu * d2
            )
            hz = 1j * u / self.r * (ju * a2 + yu * c2)
        else:
            y_tm = self.y_tm_clad(w, h)
            v = self.samples.v(h ** 2, w, e2)
            vr = v * r / self.r
            kv = ssp.kv(n, vr)
            kv_plus = ssp.kv(n + 1, vr)
            kv_minus = ssp.kv(n - 1, vr)
            nkv_v = -(kv_minus - kv_plus) / 2
            kpv = -(kv_minus + kv_plus) / 2
            er = nkv_v * a3 + kpv * b3
            ep = kpv * a3 + nkv_v * b3
            ez = -v / (1j * h * self.r) * kv * b3
            hr = -(y_te * kpv * a3 + y_tm * nkv_v * b3)
            hp = y_te * nkv_v * a3 + y_tm * kpv * b3
            hz = -1j * v / self.r * kv * a3
        return np.array([er, ep, ez, hr, hp, hz])

    def fields(self, x, y, w, dir, alpha, h, coef):
        """Return the electromagnetic field vectors for the specified mode and
        point

        Args:
            x: A float indicating the x coordinate [um]
            y: A float indicating the y coordinate [um]
            w: A complex indicating the angular frequency
            dir: "h" (horizontal polarization) or "v" (vertical polarization)
            alpha: A tuple (pol, n, m) where pol is 'M' for TM-like mode or
                'E' for TE-like mode, n is the order of the mode, and m is
                the number of modes in the order and the polarization.
            h: A complex indicating the phase constant.
            coef: The coefficients of TE- and TM- components
        Returns:
            f_vec: An array of complexes [ex, ey, ez, hx, hy, hz].
        """
        pol, n, m = alpha
        r = np.hypot(x, y)
        p = np.arctan2(y, x)
        w = w.real + 1j * w.imag
        h = h.real + 1j * h.imag
        if dir == "h":
            fr = np.cos(n * p)
            fp = -np.sin(n * p)
        else:
            fr = np.sin(n * p)
            fp = np.cos(n * p)
        er, ep, ez, hr, hp, hz = self.field_r_dep(r, w, alpha, h, coef)
        er *= fr
        ep *= fp
        ez *= fr
        hr *= fp
        hp *= fr
        hz *= fp
        ex = er * np.cos(p) - ep * np.sin(p)
        ey = er * np.sin(p) + ep * np.cos(p)
        hx = hr * np.cos(p) - hp * np.sin(p)
        hy = hr * np.sin(p) + hp * np.cos(p)
        return np.array([ex, ey, ez, hx, hy, hz])
