from __future__ import annotations

import cmath
from logging import getLogger
from typing import Dict, List, Tuple

import numpy as np

from pymwm.utils.slit_utils import ABY_cython, coefs_cython, uvABY_cython
from pymwm.waveguide import Database, Waveguide

from .samples import Samples, SamplesForRay, SamplesLowLoss, SamplesLowLossForRay

logger = getLogger(__package__)


class Slit(Waveguide):
    """A class defining a slit waveguide."""

    def __init__(self, params):
        """Init Slit class.

        Args:
            params: A dict whose keys and values are as follows:
                'core': A dict of the setting parameters of the core:
                    'shape': A string indicating the shape of the core.
                    'size': A float indicating the width of the slit [um].
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
                    'ls': A list of characters chosen from "h" (horizontal
                        polarization) and "v" (vertical polarization).
                        In the slit case, "h" ("v") corresponds to TE (TM)
                        polarization.
        """
        num_m = params["modes"].setdefault("num_m", 1)
        if num_m != 1:
            logger.warning(
                "num_m must be 1 if shape is slit." + "The set value is ignored."
            )
            params["modes"]["num_m"] = 1
        super().__init__(params)

    def get_alphas(self, alpha_list: List[Tuple[str, int, int]]) -> Dict:
        alphas = {"h": [], "v": []}
        for alpha in [("E", n, 1) for n in range(1, self.num_n)]:
            if alpha in alpha_list:
                alphas["v"].append(alpha)
        for alpha in [("M", n, 1) for n in range(self.num_n)]:
            if alpha in alpha_list:
                alphas["h"].append(alpha)
        return alphas

    def merge_even_and_odd_data(
        self, even: tuple[np.ndarray, np.ndarray], odd: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        xs_e, success_e = even
        xs_o, success_o = odd
        num_n_e = xs_e.shape[2]
        num_n_o = xs_o.shape[2]
        xs_list = []
        success_list = []
        for i in range(num_n_o):
            xs_list += [xs_e[:, :, i], xs_o[:, :, i]]
            success_list += [success_e[:, :, i], success_o[:, :, i]]
        if num_n_e > num_n_o:
            xs_list.append(xs_e[:, :, -1])
            success_list.append(success_e[:, :, -1])
        return np.dstack(xs_list), np.dstack(success_list)

    def betas_convs_samples(
        self, params: Dict
    ) -> Tuple[np.ndarray, np.ndarray, Samples]:
        im_factor = self.clad.im_factor
        self.clad.im_factor = 1.0
        p_modes = params["modes"].copy()
        num_n_0 = p_modes["num_n"]
        betas = convs = None
        success = False
        catalog = Database().load_catalog()
        num_n_max = catalog["num_n"].max()
        if not np.isnan(num_n_max):
            for num_n in [n for n in range(num_n_0, num_n_max + 1)]:
                p_modes["num_n"] = num_n
                smp = Samples(self.r, self.fill_params, self.clad_params, p_modes)
                try:
                    betas, convs = smp.database.load()
                    success = True
                    break
                except IndexError:
                    continue
        if not success:
            import ray

            p_modes["num_n"] = num_n_0
            smp = Samples(self.r, self.fill_params, self.clad_params, p_modes)
            ray.shutdown()
            try:
                ray.init()
                p_modes_id = ray.put(p_modes)
                pool = ray.util.ActorPool(
                    SamplesForRay.remote(
                        self.r, self.fill_params, self.clad_params, p_modes_id
                    )
                    for _ in range(4)
                )
                xs_success_list = list(
                    pool.map(
                        lambda a, arg: a.task.remote(arg),
                        [
                            ("M", "even", num_n_0),
                            ("M", "odd", num_n_0),
                            ("E", "even", num_n_0),
                            ("E", "odd", num_n_0),
                        ],
                    )
                )
            finally:
                ray.shutdown()
            xs_success_M = self.merge_even_and_odd_data(
                xs_success_list[0], xs_success_list[1]
            )
            xs_success_E = self.merge_even_and_odd_data(
                xs_success_list[2], xs_success_list[3]
            )
            betas, convs = smp.betas_convs([xs_success_M, xs_success_E])
            smp.database.save(betas, convs)
        if im_factor != 1.0:
            self.clad.im_factor = im_factor
            smp = SamplesLowLoss(self.r, self.fill_params, self.clad_params, p_modes)
            try:
                betas, convs = smp.database.load()
            except IndexError:
                self.clad.im_factor = im_factor
                num_n = p_modes["num_n"]
                ns = list(range(num_n))
                ns_e = ns[::2]
                ns_o = ns[1::2]
                args = []
                for iwr in range(len(smp.ws)):
                    for iwi in range(len(smp.wis)):
                        xis_list = [
                            [betas[("M", n, 1)][iwr, iwi] ** 2 for n in ns_e],
                            [betas[("M", n, 1)][iwr, iwi] ** 2 for n in ns_o],
                            [betas[("E", n, 1)][iwr, iwi] ** 2 for n in ns_e],
                            [betas[("E", n, 1)][iwr, iwi] ** 2 for n in ns_o],
                        ]
                        args.append((iwr, iwi, xis_list))
                import ray

                ray.shutdown()
                try:
                    ray.init()
                    p_modes_id = ray.put(p_modes)
                    pool = ray.util.ActorPool(
                        SamplesLowLossForRay.remote(
                            self.r, self.fill_params, self.clad_params, p_modes_id
                        )
                        for _ in range(16)
                    )
                    xs_success_list = list(
                        pool.map(lambda a, arg: a.task.remote(arg), args)
                    )

                finally:
                    ray.shutdown()
                betas, convs = smp.betas_convs(xs_success_list)
                smp.database.save(betas, convs)
        return betas, convs, smp

    def beta(self, w, alpha):
        """Return phase constant

        Args:
            w: A complex indicating the angular frequency
            alpha: A tuple (pol, n, m) where pol is 'M' for TM mode or
                'E' for TE mode, n is the order of the mode, and m is
                the number of modes in the order and the polarization,
                which is always 1 in the slit case.
        Returns:
            h: A complex indicating the phase constant.
        """
        wr = w.real
        wi = w.imag
        hr = self.beta_funcs[(alpha, "real")](wr, wi)[0, 0]
        hi = self.beta_funcs[(alpha, "imag")](wr, wi)[0, 0]
        # if hr < 0:
        #     hr = 1e-16
        # if hi < 0:
        #     hi = 1e-16
        return hr + 1j * hi

    def beta_pec(self, w, alpha):
        """Return phase constant of PEC waveguide

        Args:
            w: A complex indicating the angular frequency
            alpha: A tuple (pol, n, m) where pol is 'M' for TM mode or
                'E' for TE mode, n is the order of the mode, and m is the
                number of modes in the order and the polarization,
                which is always 1 in the slit case.
                Here, the order of TE mode starts with 1.
        Returns:
            h: A complex indicating the phase constant.
        """
        w_comp = w.real + 1j * w.imag
        pol, n, m = alpha
        val = cmath.sqrt(self.fill(w_comp) * w_comp ** 2 - (n * np.pi / self.r) ** 2)
        if abs(val.real) > abs(val.imag):
            if val.real < 0:
                val *= -1
        else:
            if val.imag < 0:
                val *= -1
        return val

    def coef(self, h, w, alpha):
        """Return the coefficients of TE- and TM- components which compose
        the hybrid mode.

        Args:
            h: A complex indicating the phase constant.
            w: A complex indicating the angular frequency
            alpha: A tuple (pol, n, m) where pol is 'M' for TM mode or
                'E' for TE mode, n is the order of the mode, and m is
                the number of modes in the order and the polarization,
                which is always 1 in the slit case.
        Returns:
            a: A complex indicating the coefficient of TE-component
            b: A complex indicating the coefficient of TM-component
        """
        pol, n, m = alpha
        w = w.real + 1j * w.imag
        h = h.real + 1j * h.imag
        if pol == "E":
            norm = self.norm(w, h, alpha, 1.0 + 0.0j, 0.0j)
            ai, bi = 1.0 / norm, 0.0
        else:
            norm = self.norm(w, h, alpha, 0.0j, 1.0 + 0.0j)
            ai, bi = 0.0, 1.0 / norm
        return ai, bi

    @staticmethod
    def sinc(x):
        x1 = x / np.pi
        return np.sinc(x1)

    def norm(self, w, h, alpha, a, b):
        a2_b2 = a ** 2 + b ** 2
        e1 = self.fill(w)
        e2 = self.clad(w)
        pol, n, m = alpha
        if self.clad(w).real < -1e6:
            if pol == "M" and n == 0:
                return cmath.sqrt(a2_b2 * self.r)
            else:
                return cmath.sqrt(a2_b2 * self.r / 2)
        u = self.samples.u(h ** 2, w, e1)
        # uc = u.conjugate()
        v = self.samples.v(h ** 2, w, e2)
        # vc = v.conjugate()
        if n % 2 == 0:
            if pol == "E":
                b_a = cmath.sin(u)
                parity = -1
            else:
                b_a = u / v * cmath.sin(u)
                parity = 1
        else:
            if pol == "E":
                b_a = cmath.cos(u)
                parity = 1
            else:
                b_a = -u / v * cmath.cos(u)
                parity = -1
        val = cmath.sqrt(
            a2_b2
            * self.r
            * (b_a ** 2 / (2 * v) + (1.0 + parity * self.sinc(2 * u)) / 2)
        )
        return val

    def Y(self, w, h, alpha, a, b):
        """Return the effective admittance of the waveguide mode

        Args:
            w: A complex indicating the angular frequency
            h: A complex indicating the phase constant.
            alpha: A tuple (pol, n, m) where pol is 'M' for TM mode or
                'E' for TE mode, n is the order of the mode, and m is
                the number of modes in the order and the polarization,
                which is always 1 in the slit case.
            a: A complex indicating the coefficient of TE-component
            b: A complex indicating the coefficient of TM-component
        Returns:
            y: A complex indicating the effective admittance
        """
        pol, n, m = alpha
        e1 = self.fill(w)
        e2 = self.clad(w)
        y_te = self.y_te(w, h)
        y_tm_in = self.y_tm_inner(w, h)
        y_tm_out = self.y_tm_outer(w, h)
        if e2.real < -1e6:
            if pol == "E":
                return y_te
            else:
                return y_tm_in
        u = self.samples.u(h ** 2, w, e1)
        v = self.samples.v(h ** 2, w, e2)
        if pol == "E":
            y_in = y_out = y_te
        else:
            y_in = y_tm_in
            y_out = y_tm_out
        if n % 2 == 0:
            if pol == "E":
                b_a = np.sin(u)
                parity = -1
            else:
                b_a = u / v * np.sin(u)
                parity = 1
        else:
            if pol == "E":
                b_a = np.cos(u)
                parity = 1
            else:
                b_a = -u / v * np.cos(u)
                parity = -1
        val = (
            (a ** 2 + b ** 2)
            * self.r
            * (
                y_out * b_a ** 2 / (2 * v)
                + (1.0 + parity * self.sinc(2 * u)) * y_in / 2
            )
        )
        return val

    def Yab(self, w, h1, s1, l1, n1, m1, a1, b1, h2, s2, l2, n2, m2, a2, b2):
        """Return the admittance matrix element of the waveguide modes

        Args:
            w: A complex indicating the angular frequency
            h1, h2: A complex indicating the phase constant.
            s1, s2: 0 for TE mode or 1 for TM mode
            l1, l2: 0 for h mode or 1 for v mode. In the slit case,
                l=h for TM mode, and l=v for TE mode.
            n1, n2: the order of the mode
            m1, m2: the number of modes in the order and the polarization.
                They are always 1 in the slit case.
            a1, a2: A complex indicating the coefficient of TE-component
            b1, b2: A complex indicating the coefficient of TM-component
        Returns:
            y: A complex indicating the effective admittance
        """
        if s1 != s2:
            return 0.0
        if n1 % 2 != n2 % 2:
            return 0.0
        e1 = self.fill(w)
        e2 = self.clad(w)
        y_te = self.y_te(w, h2)
        y_tm_in = self.y_tm_inner(w, h2)
        y_tm_out = self.y_tm_outer(w, h2)
        if e2.real < -1e6:
            if n1 != n2:
                return 0.0
            if s1 == 0:
                return y_te
            else:
                return y_tm_in
        ac = a1
        a = a2
        bc = b1
        b = b2
        uc = self.samples.u(h1 ** 2, w, e1)
        u = self.samples.u(h2 ** 2, w, e1)
        vc = self.samples.v(h1 ** 2, w, e2)
        v = self.samples.v(h2 ** 2, w, e2)
        if s1 == 0:
            y_in = y_out = y_te
            val = ac * a * self.r
        else:
            y_in = y_tm_in
            y_out = y_tm_out
            val = bc * b * self.r
        if n1 % 2 == 0:
            if s1 == 0:
                b_ac = np.sin(uc)
                b_a = np.sin(u)
                parity = -1
            else:
                b_ac = uc / vc * np.sin(uc)
                b_a = u / v * np.sin(u)
                parity = 1
        else:
            if s1 == 0:
                b_ac = np.cos(uc)
                b_a = np.cos(u)
                parity = 1
            else:
                b_ac = -uc / vc * np.cos(uc)
                b_a = -u / v * np.cos(u)
                parity = -1
        val *= (
            y_out * b_ac * b_a / (v + vc)
            + y_in * (self.sinc(u - uc) + parity * self.sinc(u + uc)) / 2
        )
        return val

    @staticmethod
    def y_te(w, h):
        return h / w

    def y_tm_inner(self, w, h):
        e = self.fill(w)
        return e * w / h

    def y_tm_outer(self, w, h):
        e = self.clad(w)
        return e * w / h

    def fields(self, x, y, w, dir, alpha, h, coef):
        """Return the electromagnetic field vectors for the specified mode and
        point

        Args:
            x: A float indicating the x coordinate [um]
            y: A float indicating the y coordinate [um]
            w: A complex indicating the angular frequency
            dir: "h" (horizontal polarization) or "v" (vertical polarization).
                In the slit case, dir='h' for TM and dir='v' for TE.
            alpha: A tuple (pol, n, m) where pol is 'M' for TM-like mode or
                'E' for TE-like mode, n is the order of the mode, and m is
                the number of modes in the order and the polarization,
                which is always 1 in the slit case.
            h: A complex indicating the phase constant.
            coef: The coefficients of TE- and TM- components
        Returns:
            f_vec: An array of complexes [ex, ey, ez, hx, hy, hz].
        """
        pol, n, m = alpha
        a, b = coef
        u = self.samples.u(h ** 2, w, self.fill(w))
        v = self.samples.v(h ** 2, w, self.clad(w))
        gd = u / (self.r / 2)
        gm = v / (self.r / 2)
        if pol == "E":
            y_te = self.y_te(w, h)
            ex = ez = 0.0
            hy = 0.0
            if n % 2 == 1:
                # parity even
                if abs(x) <= self.r / 2:
                    ey = a * np.cos(gd * x)
                    hx = y_te * ey
                    hz = 1j * gd / w * a * np.sin(gd * x)
                else:
                    b_a = np.exp(v) * np.cos(u)
                    ey = a * b_a * np.exp(-gm * abs(x))
                    hx = y_te * ey
                    hz = 1j * gm / w * x / abs(x) * ey
            else:
                # parity odd
                if abs(x) <= self.r / 2:
                    ey = a * np.sin(gd * x)
                    hx = y_te * ey
                    hz = -1j * gd / w * a * np.cos(gd * x)
                else:
                    b_a = np.exp(v) * np.sin(u)
                    ey = a * b_a * x / abs(x) * np.exp(-gm * abs(x))
                    hx = y_te * ey
                    hz = 1j * gm / w * x / abs(x) * ey
        else:
            hx = hz = 0.0
            ey = 0.0
            if n % 2 == 0:
                # parity even
                if abs(x) <= self.r / 2:
                    y_tm = self.y_tm_inner(w, h)
                    ex = b * np.cos(gd * x)
                    hy = y_tm * ex
                    ez = -1j * gd / h * b * np.sin(gd * x)
                else:
                    y_tm = self.y_tm_outer(w, h)
                    b_a = u / v * np.exp(v) * np.sin(u)
                    ex = b * b_a * np.exp(-gm * abs(x))
                    hy = y_tm * ex
                    ez = -1j * gm * x / abs(x) / h * ex
            else:
                # parity odd
                if abs(x) <= self.r / 2:
                    y_tm = self.y_tm_inner(w, h)
                    ex = b * np.sin(gd * x)
                    hy = y_tm * ex
                    ez = 1j * gd / h * b * np.cos(gd * x)
                else:
                    y_tm = self.y_tm_outer(w, h)
                    b_a = -u / v * np.exp(v) * np.cos(u)
                    ex = b * b_a * x / abs(x) * np.exp(-gm * abs(x))
                    hy = y_tm * ex
                    ez = -1j * gm * x / abs(x) / h * ex
        return np.array([ex, ey, ez, hx, hy, hz])

    def e_field(self, x, y, w, dir, alpha, h, coef) -> np.ndarray:
        """Return the electric field vector for the specified mode and
        point

        Args:
            x: A float indicating the x coordinate [um]
            y: A float indicating the y coordinate [um]
            w: A complex indicating the angular frequency
            dir: "h" (horizontal polarization) or "v" (vertical polarization).
                In the slit case, dir='h' for TM and dir='v' for TE.
            alpha: A tuple (pol, n, m) where pol is 'M' for TM mode or
                'E' for TE mode, n is the order of the mode, and m is
                the number of modes in the order and the polarization,
                which is always 1 in the slit case.
            h: A complex indicating the phase constant.
            coef: The coefficients of TE- and TM- components
        Returns:
            e_vec: Array(ex, ey, ez).
        """
        pol, n, m = alpha
        a, b = coef
        u = self.samples.u(h ** 2, w, self.fill(w))
        v = self.samples.v(h ** 2, w, self.clad(w))
        gd = u / (self.r / 2)
        gm = v / (self.r / 2)
        if pol == "E":
            ex = ez = 0.0
            if n % 2 == 1:
                # parity even
                if abs(x) <= self.r / 2:
                    ey = a * np.cos(gd * x)
                else:
                    b_a = np.exp(v) * np.cos(u)
                    ey = a * b_a * np.exp(-gm * abs(x))
            else:
                # parity odd
                if abs(x) <= self.r / 2:
                    ey = a * np.sin(gd * x)
                else:
                    b_a = np.exp(v) * np.sin(u)
                    ey = a * b_a * x / abs(x) * np.exp(-gm * abs(x))
        else:
            ey = 0.0
            if n % 2 == 0:
                # parity even
                if abs(x) <= self.r / 2:
                    ex = b * np.cos(gd * x)
                    ez = -1j * gd / h * b * np.sin(gd * x)
                else:
                    b_a = u / v * np.exp(v) * np.sin(u)
                    ex = b * b_a * np.exp(-gm * abs(x))
                    ez = -1j * gm * x / abs(x) / h * b * b_a * np.exp(-gm * abs(x))
            else:
                # parity odd
                if abs(x) <= self.r / 2:
                    ex = b * np.sin(gd * x)
                    ez = 1j * gd / h * b * np.cos(gd * x)
                else:
                    b_a = -u / v * np.exp(v) * np.cos(u)
                    ex = b * b_a * x / abs(x) * np.exp(-gm * abs(x))
                    ez = -1j * gm / h * b * b_a * np.exp(-gm * abs(x))
        return np.array([ex, ey, ez])

    def h_field(self, x, y, w, dir, alpha, h, coef) -> np.ndarray:
        """Return the magnetic field vectors for the specified mode and
        point

        Args:
            x: A float indicating the x coordinate [um]
            y: A float indicating the y coordinate [um]
            w: A complex indicating the angular frequency
            dir: "h" (horizontal polarization) or "v" (vertical polarization).
                In the slit case, dir='h' for TM and dir='v' for TE.
            alpha: A tuple (pol, n, m) where pol is 'M' for TM mode or
                'E' for TE mode, n is the order of the mode, and m is
                the number of modes in the order and the polarization,
                which is always 1 in the slit case.
            h: A complex indicating the phase constant.
            coef: The coefficients of TE- and TM- components
        Returns:
            h_vec: Array(hx, hy, hz).
        """
        pol, n, m = alpha
        a, b = coef
        u = self.samples.u(h ** 2, w, self.fill(w))
        v = self.samples.v(h ** 2, w, self.clad(w))
        gd = u / (self.r / 2)
        gm = v / (self.r / 2)
        if pol == "E":
            y_te = self.y_te(w, h)
            hy = 0.0
            if n % 2 == 1:
                # parity even
                if abs(x) <= self.r / 2:
                    hx = y_te * a * np.cos(gd * x)
                    hz = 1j * gd / w * a * np.sin(gd * x)
                else:
                    b_a = np.exp(v) * np.cos(u)
                    hx = y_te * a * b_a * np.exp(-gm * abs(x))
                    hz = 1j * gm / w * x / abs(x) * a * b_a * np.exp(-gm * abs(x))
            else:
                # parity odd
                if abs(x) <= self.r / 2:
                    hx = y_te * a * np.sin(gd * x)
                    hz = -1j * gd / w * a * np.cos(gd * x)
                else:
                    b_a = np.exp(v) * np.sin(u)
                    hx = y_te * a * b_a * x / abs(x) * np.exp(-gm * abs(x))
                    hz = (
                        1j
                        * gm
                        / w
                        * x
                        / abs(x)
                        * a
                        * b_a
                        * x
                        / abs(x)
                        * np.exp(-gm * abs(x))
                    )
        else:
            hx = hz = 0.0
            if n % 2 == 0:
                # parity even
                if abs(x) <= self.r / 2:
                    y_tm = self.y_tm_inner(w, h)
                    hy = y_tm * b * np.cos(gd * x)
                else:
                    y_tm = self.y_tm_outer(w, h)
                    b_a = u / v * np.exp(v) * np.sin(u)
                    hy = y_tm * b * b_a * np.exp(-gm * abs(x))
            else:
                # parity odd
                if abs(x) <= self.r / 2:
                    y_tm = self.y_tm_inner(w, h)
                    hy = y_tm * b * np.sin(gd * x)
                else:
                    y_tm = self.y_tm_outer(w, h)
                    b_a = -u / v * np.exp(v) * np.cos(u)
                    hy = y_tm * b * b_a * x / abs(x) * np.exp(-gm * abs(x))
        return np.array([hx, hy, hz])

    def coefs_numpy(self, hs, w):
        As = []
        Bs = []
        for h, s, n, m in zip(hs, self.s_all, self.n_all, self.m_all):
            pol = "E" if s == 0 else "M"
            ai, bi = self.coef(h, w, (pol, n, m))
            As.append(ai)
            Bs.append(bi)
        return np.ascontiguousarray(As), np.ascontiguousarray(Bs)

    def coefs(self, hs, w):
        return coefs_cython(self, hs, w)

    def Ys(self, w, hs, As, Bs):
        vals = []
        for h, s, n, a, b in zip(hs, self.s_all, self.n_all, As, Bs):
            pol = "E" if s == 0 else "M"
            vals.append(self.Y(w, h, (pol, n, 1), a, b))
        return np.array(vals)

    def hAB(self, w):
        hs = np.array([self.beta(w, alpha) for alpha in self.alpha_all])
        As, Bs = self.coefs(hs, w)
        return hs, As, Bs

    def ABY(self, w, hs):
        e1 = self.fill(w)
        e2 = self.clad(w)
        return ABY_cython(w, self.r, self.s_all, self.n_all, hs, e1, e2)

    def hABY(self, w):
        e1 = self.fill(w)
        e2 = self.clad(w)
        hs = np.array([self.beta(w, alpha) for alpha in self.alpha_all])
        As, Bs, Y = ABY_cython(w, self.r, self.s_all, self.n_all, hs, e1, e2)
        return hs, As, Bs, Y

    def huvABY(self, w):
        e1 = self.fill(w)
        e2 = self.clad(w)
        hs = np.array([self.beta(w, alpha) for alpha in self.alpha_all])
        us, vs, As, Bs, Y = uvABY_cython(w, self.r, self.s_all, self.n_all, hs, e1, e2)
        return hs, us, vs, As, Bs, Y
