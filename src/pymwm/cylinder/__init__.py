from __future__ import annotations

import cmath

import numpy as np
import psutil
import ray
import scipy.special as ssp

from pymwm.utils import cylinder_utils
from pymwm.waveguide import Database, Sampling, Waveguide

from .samples import Samples, SamplesForRay, SamplesLowLoss, SamplesLowLossForRay


class Cylinder(Waveguide):
    """A class defining a cylindrical waveguide."""

    def __init__(self, params):
        """Init Cylinder class.

        Args:
            params: A dict whose keys and values are as follows:
                'core': A dict of the setting parameters of the core:
                    'shape': A string indicating the shape of the core.
                    'size': A float indicating the radius of the circular cross
                        section [um].
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
        super().__init__(params)
        self.u_pec, self.jnu_pec, self.jnpu_pec = self.u_jnu_jnpu_pec(
            self.num_n, self.num_m
        )

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

    def betas_convs_samples(self, params: dict) -> tuple[dict, dict, Sampling]:
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
                smp = Samples(self.r, self.fill_params, self.clad_params, p_modes)
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
        smp = Samples(self.r, self.fill_params, self.clad_params, p_modes)
        ray.shutdown()
        try:
            ray.init()
            p_modes_id = ray.put(p_modes)
            pool = ray.util.ActorPool(
                SamplesForRay.remote(
                    self.r, self.fill_params, self.clad_params, p_modes_id
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
        smp = SamplesLowLoss(self.r, self.fill_params, self.clad_params, p_modes)
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
                        self.r, self.fill_params, self.clad_params, p_modes_id
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
        wr = w.real
        wi = w.imag
        hr: float = self.beta_funcs[(alpha, "real")](wr, wi)[0, 0]
        hi: float = self.beta_funcs[(alpha, "imag")](wr, wi)[0, 0]
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
                number of modes in the order and the polarization.
        Returns:
            h: A complex indicating the phase constant.
        """
        w_comp = w.real + 1j * w.imag
        pol, n, m = alpha
        if pol == "E":
            chi = ssp.jnp_zeros(n, m)[-1]
        elif pol == "M":
            chi = ssp.jn_zeros(n, m)[-1]
        else:
            raise ValueError("pol must be 'E' or 'M")
        val = cmath.sqrt(self.fill(w_comp) * w_comp ** 2 - chi ** 2 / self.r ** 2)
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
            alpha: A tuple (pol, n, m) where pol is 'M' for TM-like mode or
                'E' for TE-like mode, n is the order of the mode, and m is
                the number of modes in the order and the polarization.
        Returns:
            a: A complex indicating the coefficient of TE-component
            b: A complex indicating the coefficient of TM-component
        """
        e1 = self.fill(w)
        e2 = self.clad(w)
        pol, n, m = alpha
        w = w.real + 1j * w.imag
        h = h.real + 1j * h.imag
        if e2.real < -1e6:
            if pol == "E":
                norm = self.norm(w, h, alpha, 1.0 + 0.0j, 0.0j)
                ai, bi = 1.0 / norm, 0.0
            else:
                norm = self.norm(w, h, alpha, 0.0j, 1.0 + 0.0j)
                ai, bi = 0.0, 1.0 / norm
        else:
            u = self.samples.u(h ** 2, w, e1)
            v = self.samples.v(h ** 2, w, e2)
            knv = ssp.kv(n, v)
            knpv = ssp.kvp(n, v)
            jnu = ssp.jv(n, u)
            jnpu = ssp.jvp(n, u)
            ci = -n * (u ** 2 + v ** 2) * jnu * knv / (u * v)
            if pol == "E":
                ci *= (h / w) ** 2
                ci /= e1 * jnpu * v * knv + e2 * knpv * u * jnu
                norm = self.norm(w, h, alpha, 1.0 + 0.0j, ci)
                ai = 1.0 / norm
                bi = ci / norm
            else:
                ci /= jnpu * v * knv + knpv * u * jnu
                norm = self.norm(w, h, alpha, ci, 1.0 + 0.0j)
                bi = 1.0 / norm
                ai = ci / norm
        return ai, bi

    def norm(self, w, h, alpha, a, b):
        pol, n, m = alpha
        en = 1 if n == 0 else 2
        if self.clad(w).real < -1e6:
            radius = self.r
            if pol == "E":
                u = ssp.jnp_zeros(n, m)[-1]
                jnu = ssp.jv(n, u)
                jnpu = 0.0
            else:
                u = ssp.jn_zeros(n, m)[-1]
                jnu = 0.0
                jnpu = ssp.jvp(n, u)
            return cmath.sqrt(
                a ** 2 * np.pi * radius ** 2 / en * (1 - n ** 2 / u ** 2) * jnu ** 2
                + b ** 2 * np.pi * radius ** 2 / en * jnpu ** 2
            )
        u = self.samples.u(h ** 2, w, self.fill(w))
        jnu = ssp.jv(n, u)
        jnpu = ssp.jvp(n, u)
        v = self.samples.v(h ** 2, w, self.clad(w))
        knv = ssp.kv(n, v)
        knpv = ssp.kvp(n, v)
        val_u = 2 * np.pi * self.r ** 2 / en
        val_v = val_u * ((u * jnu) / (v * knv)) ** 2
        upart_diag = self.upart_diag(n, u, jnu, jnpu)
        vpart_diag = self.vpart_diag(n, v, knv, knpv)
        upart_off = self.upart_off(n, u, jnu)
        vpart_off = self.vpart_off(n, v, knv)
        return cmath.sqrt(
            val_u
            * (
                a * (a * upart_diag + b * upart_off)
                + b * (b * upart_diag + a * upart_off)
            )
            - val_v
            * (
                a * (a * vpart_diag + b * vpart_off)
                + b * (b * vpart_diag + a * vpart_off)
            )
        )

    @staticmethod
    def upart_diag(n, u, jnu, jnpu):
        return jnu * jnpu / u + (jnpu ** 2 + (1 - n ** 2 / u ** 2) * jnu ** 2) / 2

    @staticmethod
    def upart_off(n, u, jnu):
        return n * (jnu / u) ** 2

    @staticmethod
    def vpart_diag(n, v, knv, knpv):
        return knv * knpv / v + (knpv ** 2 - (1 + n ** 2 / v ** 2) * knv ** 2) / 2

    @staticmethod
    def vpart_off(n, v, knv):
        return n * (knv / v) ** 2

    def Y(self, w, h, alpha, a, b):
        """Return the effective admittance of the waveguide mode

        Args:
            w: A complex indicating the angular frequency
            h: A complex indicating the phase constant.
            alpha: A tuple (pol, n, m) where pol is 'M' for TM-like mode or
                'E' for TE-like mode, n is the order of the mode, and m is
                the number of modes in the order and the polarization.
            a: A complex indicating the coefficient of TE-component
            b: A complex indicating the coefficient of TM-component
        Returns:
            y: A complex indicating the effective admittance
        """
        pol, n, m = alpha
        e1 = self.fill(w)
        e2 = self.clad(w)
        en = 1 if n == 0 else 2
        if e2.real < -1e6:
            if pol == "E":
                val = h / w
            else:
                val = e1 * w / h
        else:
            u = self.samples.u(h ** 2, w, e1)
            jnu = ssp.jv(n, u)
            jnpu = ssp.jvp(n, u)
            v = self.samples.v(h ** 2, w, e2)
            knv = ssp.kv(n, v)
            knpv = ssp.kvp(n, v)
            val_u = 2 * np.pi * self.r ** 2 / en
            val_v = val_u * ((u * jnu) / (v * knv)) ** 2
            upart_diag = self.upart_diag(n, u, jnu, jnpu)
            vpart_diag = self.vpart_diag(n, v, knv, knpv)
            upart_off = self.upart_off(n, u, jnu)
            vpart_off = self.vpart_off(n, v, knv)
            val = val_u * (
                h / w * a * (a * upart_diag + b * upart_off)
                + e1 * w / h * b * (b * upart_diag + a * upart_off)
            ) - val_v * (
                h / w * a * (a * vpart_diag + b * vpart_off)
                + e2 * w / h * b * (b * vpart_diag + a * vpart_off)
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
        a, b = coef
        r = np.hypot(x, y)
        p = np.arctan2(y, x)
        u = self.samples.u(h ** 2, w, self.fill(w))
        v = self.samples.v(h ** 2, w, self.clad(w))
        ur = u * r / self.r
        vr = v * r / self.r
        if dir == "h":
            fr = np.cos(n * p)
            fp = -np.sin(n * p)
        else:
            fr = np.sin(n * p)
            fp = np.cos(n * p)
        y_te = self.y_te(w, h)
        if r <= self.r:
            y_tm = self.y_tm_inner(w, h)
            er_te = (ssp.jv(n - 1, ur) + ssp.jv(n + 1, ur)) / 2 * fr
            er_tm = ssp.jvp(n, ur) * fr
            er = a * er_te + b * er_tm
            ep_te = ssp.jvp(n, ur) * fp
            ep_tm = (ssp.jv(n - 1, ur) + ssp.jv(n + 1, ur)) / 2 * fp
            ep = a * ep_te + b * ep_tm
            ez = u / (1j * h * self.r) * b * ssp.jv(n, ur) * fr
            hr = -y_te * a * ep_te - y_tm * b * ep_tm
            hp = y_te * a * er_te + y_tm * b * er_tm
            hz = -u / (1j * h * self.r) * y_te * a * ssp.jv(n, ur) * fp
        else:
            y_tm = self.y_tm_outer(w, h)
            val = -u * ssp.jv(n, u) / (v * ssp.kv(n, v))
            er_te = -(ssp.kv(n - 1, vr) - ssp.kv(n + 1, vr)) / 2 * fr * val
            er_tm = ssp.kvp(n, vr) * fr * val
            er = a * er_te + b * er_tm
            ep_te = ssp.kvp(n, vr) * fp * val
            ep_tm = -(ssp.kv(n - 1, vr) - ssp.kv(n + 1, vr)) / 2 * fp * val
            ep = a * ep_te + b * ep_tm
            ez = -v / (1j * h * self.r) * b * ssp.kv(n, vr) * fr * val
            hr = -y_te * a * ep_te - y_tm * b * ep_tm
            hp = y_te * a * er_te + y_tm * b * er_tm
            hz = v / (1j * h * self.r) * y_te * a * ssp.kv(n, vr) * fp * val
        ex = er * np.cos(p) - ep * np.sin(p)
        ey = er * np.sin(p) + ep * np.cos(p)
        hx = hr * np.cos(p) - hp * np.sin(p)
        hy = hr * np.sin(p) + hp * np.cos(p)
        return np.array([ex, ey, ez, hx, hy, hz])

    def e_field(self, x, y, w, dir, alpha, h, coef):
        """Return the electric field vector for the specified mode and
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
            e_vec: An array of complexes [ex, ey, ez].
        """
        pol, n, m = alpha
        a, b = coef
        r = np.hypot(x, y)
        p = np.arctan2(y, x)
        u = self.samples.u(h ** 2, w, self.fill(w))
        v = self.samples.v(h ** 2, w, self.clad(w))
        ur = u * r / self.r
        vr = v * r / self.r
        if dir == "h":
            fr = np.cos(n * p)
            fp = -np.sin(n * p)
        else:
            fr = np.sin(n * p)
            fp = np.cos(n * p)
        if r <= self.r:
            er_te = (ssp.jv(n - 1, ur) + ssp.jv(n + 1, ur)) / 2 * fr
            er_tm = ssp.jvp(n, ur) * fr
            er = a * er_te + b * er_tm
            ep_te = ssp.jvp(n, ur) * fp
            ep_tm = (ssp.jv(n - 1, ur) + ssp.jv(n + 1, ur)) / 2 * fp
            ep = a * ep_te + b * ep_tm
            ez = u / (1j * h * self.r) * b * ssp.jv(n, ur) * fr
        else:
            val = -u * ssp.jv(n, u) / (v * ssp.kv(n, v))
            er_te = -(ssp.kv(n - 1, vr) - ssp.kv(n + 1, vr)) / 2 * fr * val
            er_tm = ssp.kvp(n, vr) * fr * val
            er = a * er_te + b * er_tm
            ep_te = ssp.kvp(n, vr) * fp * val
            ep_tm = -(ssp.kv(n - 1, vr) - ssp.kv(n + 1, vr)) / 2 * fp * val
            ep = a * ep_te + b * ep_tm
            ez = -v / (1j * h * self.r) * b * ssp.kv(n, vr) * fr * val
        ex = er * np.cos(p) - ep * np.sin(p)
        ey = er * np.sin(p) + ep * np.cos(p)
        return np.array([ex, ey, ez])

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
        a, b = coef
        r = np.hypot(x, y)
        p = np.arctan2(y, x)
        u = self.samples.u(h ** 2, w, self.fill(w))
        v = self.samples.v(h ** 2, w, self.clad(w))
        ur = u * r / self.r
        vr = v * r / self.r
        if dir == "h":
            fr = np.cos(n * p)
            fp = -np.sin(n * p)
        else:
            fr = np.sin(n * p)
            fp = np.cos(n * p)
        y_te = self.y_te(w, h)
        if r <= self.r:
            y_tm = self.y_tm_inner(w, h)
            er_te = (ssp.jv(n - 1, ur) + ssp.jv(n + 1, ur)) / 2 * fr
            er_tm = ssp.jvp(n, ur) * fr
            ep_te = ssp.jvp(n, ur) * fp
            ep_tm = (ssp.jv(n - 1, ur) + ssp.jv(n + 1, ur)) / 2 * fp
            hr = -y_te * a * ep_te - y_tm * b * ep_tm
            hp = y_te * a * er_te + y_tm * b * er_tm
            hz = -u / (1j * h * self.r) * y_te * a * ssp.jv(n, ur) * fp
        else:
            y_tm = self.y_tm_outer(w, h)
            val = -u * ssp.jv(n, u) / (v * ssp.kv(n, v))
            er_te = -(ssp.kv(n - 1, vr) - ssp.kv(n + 1, vr)) / 2 * fr * val
            er_tm = ssp.kvp(n, vr) * fr * val
            ep_te = ssp.kvp(n, vr) * fp * val
            ep_tm = -(ssp.kv(n - 1, vr) - ssp.kv(n + 1, vr)) / 2 * fp * val
            hr = -y_te * a * ep_te - y_tm * b * ep_tm
            hp = y_te * a * er_te + y_tm * b * er_tm
            hz = v / (1j * h * self.r) * y_te * a * ssp.kv(n, vr) * fp * val
        hx = hr * np.cos(p) - hp * np.sin(p)
        hy = hr * np.sin(p) + hp * np.cos(p)
        return np.array([hx, hy, hz])

    @staticmethod
    def u_jnu_jnpu_pec(num_n, num_m):
        us = np.empty((2, num_n, num_m))
        jnus = np.empty((2, num_n, num_m))
        jnpus = np.empty((2, num_n, num_m))
        for n in range(num_n):
            us[0, n] = ssp.jnp_zeros(n, num_m)
            us[1, n] = ssp.jn_zeros(n, num_m)
            jnus[0, n] = ssp.jv(n, us[0, n])
            jnus[1, n] = np.zeros(num_m)
            jnpus[0, n] = np.zeros(num_m)
            jnpus[1, n] = ssp.jvp(n, us[1, n])
        return us, jnus, jnpus

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
        return cylinder_utils.coefs_cython(self, hs, w)

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
        return cylinder_utils.ABY_cython(
            w,
            self.r,
            self.s_all,
            self.n_all,
            self.m_all,
            hs,
            e1,
            e2,
            self.u_pec,
            self.jnu_pec,
            self.jnpu_pec,
        )

    def hABY(self, w):
        e1 = self.fill(w)
        e2 = self.clad(w)
        hs = np.array([self.beta(w, alpha) for alpha in self.alpha_all])
        As, Bs, Y = cylinder_utils.ABY_cython(
            w,
            self.r,
            self.s_all,
            self.n_all,
            self.m_all,
            hs,
            e1,
            e2,
            self.u_pec,
            self.jnu_pec,
            self.jnpu_pec,
        )
        return hs, As, Bs, Y

    def huvABY(self, w):
        e1 = self.fill(w)
        e2 = self.clad(w)
        hs = np.array([self.beta(w, alpha) for alpha in self.alpha_all])
        us, vs, As, Bs, Y = cylinder_utils.uvABY_cython(
            w,
            self.r,
            self.s_all,
            self.n_all,
            self.m_all,
            hs,
            e1,
            e2,
            self.u_pec,
            self.jnu_pec,
            self.jnpu_pec,
        )
        return hs, us, vs, As, Bs, Y
