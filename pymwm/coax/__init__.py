# -*- coding: utf-8 -*-
from typing import Dict, List, Tuple

import numpy as np
from scipy.special import jn_zeros, jnp_zeros, jv, jvp, kv, kvp

from pymwm.waveguide import Waveguide

from .samples import Samples, SamplesLowLoss


class Coax(Waveguide):
    """A class defining a coax waveguide."""

    def __init__(self, params):
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

    def get_alphas(self, alpha_list: List[Tuple[str, int, int]]) -> Dict:
        alphas = {"h": [], "v": []}
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

    def betas_convs_samples(
        self, params: Dict
    ) -> Tuple[np.ndarray, np.ndarray, Samples]:
        im_factor = 1.0
        if self.clad.im_factor != 1.0:
            im_factor = self.clad.im_factor
            self.clad.im_factor = 1.0
        p_modes = params["modes"].copy()
        num_n_0 = p_modes["num_n"]
        num_m_0 = p_modes["num_m"]
        smp = Samples(self.r, self.fill, self.clad, p_modes, self.ri)
        try:
            betas, convs = smp.database.load()
            success = True
        except IndexError:
            betas = convs = None
            success = False
        for num_n, num_m in [
            (n, m) for n in range(num_n_0, 17) for m in range(num_m_0, 5)
        ]:
            if (num_n, num_m) == (num_n_0, num_m_0):
                if success:
                    break
                else:
                    continue
            p_modes["num_n"] = num_n
            p_modes["num_m"] = num_m
            smp = Samples(self.r, self.fill, self.clad, p_modes, self.ri)
            try:
                betas, convs = smp.database.load()
                success = True
                break
            except IndexError:
                continue
        if not success:
            p_modes["num_n"] = num_n_0
            p_modes["num_m"] = num_m_0
            smp = Samples(self.r, self.fill, self.clad, p_modes, self.ri)
            from multiprocessing import Pool

            p = Pool(num_n_0)
            xs_success_list = p.map(smp, range(num_n_0))
            # betas_list = list(map(smp, range(num_n)))
            # betas = {key: val for betas, convs in betas_list
            #          for key, val in betas.items()}
            # convs = {key: val for betas, convs in betas_list
            #          for key, val in convs.items()}
            betas, convs = smp.betas_convs(xs_success_list)
            smp.database.save(betas, convs)
        if im_factor != 1.0:
            self.clad.im_factor = im_factor
            smp = SamplesLowLoss(self.r, self.fill, self.clad, p_modes, self.ri)
            try:
                betas, convs = smp.database.load()
            except IndexError:
                self.clad.im_factor = im_factor
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
                from multiprocessing import Pool

                p = Pool(16)
                xs_success_list = p.map(smp, args)
                # xs_success_list = list(map(smp, smp.args)
                betas, convs = smp.betas_convs(xs_success_list)
                smp.database.save(betas, convs)
        return betas, convs, smp

    def beta(self, w: complex, alpha: Tuple[str, int, int]) -> complex:
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
                number of modes in the order and the polarization.
        Returns:
            h: A complex indicating the phase constant.
        """
        w_comp = w.real + 1j * w.imag
        pol, n, m = alpha
        if pol == "E":
            chi = jnp_zeros(n, m)[-1]
        elif pol == "M":
            chi = jn_zeros(n, m)[-1]
        else:
            raise ValueError("pol must be 'E' or 'M'")
        val = np.sqrt(self.fill(w_comp) * w_comp ** 2 - chi ** 2 / self.r ** 2)
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
            knv = kv(n, v)
            knpv = kvp(n, v)
            jnu = jv(n, u)
            jnpu = jvp(n, u)
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
                u = jnp_zeros(n, m)[-1]
                jnu = jv(n, u)
                jnpu = 0.0
            else:
                u = jn_zeros(n, m)[-1]
                jnu = 0.0
                jnpu = jvp(n, u)
            return np.sqrt(
                a ** 2 * np.pi * radius ** 2 / en * (1 - n ** 2 / u ** 2) * jnu ** 2
                + b ** 2 * np.pi * radius ** 2 / en * jnpu ** 2
            )
        # ac = a.conjugate()
        # bc = b.conjugate()
        u = self.samples.u(h ** 2, w, self.fill(w))
        jnu = jv(n, u)
        jnpu = jvp(n, u)
        v = self.samples.v(h ** 2, w, self.clad(w))
        knv = kv(n, v)
        knpv = kvp(n, v)
        # uc = u.conjugate()
        # jnuc = jnu.conjugate()
        # jnpuc = jnpu.conjugate()
        # vc = v.conjugate()
        # knvc = knv.conjugate()
        # knpvc = knpv.conjugate()
        val_u = 2 * np.pi * self.r ** 2 / en
        val_v = val_u * ((u * jnu) / (v * knv)) ** 2
        # val_v = val_u * (uc * u * jnuc * jnu) / (vc * v * knvc * knv)
        upart_diag = self.upart_diag(n, u, jnu, jnpu, u, jnu, jnpu)
        vpart_diag = self.vpart_diag(n, v, knv, knpv, v, knv, knpv)
        upart_off = self.upart_off(n, u, jnu, u, jnu)
        vpart_off = self.vpart_off(n, v, knv, v, knv)
        return np.sqrt(
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
        # upart_diag = self.upart_diag(n, uc, jnuc, jnpuc, u, jnu, jnpu)
        # vpart_diag = self.vpart_diag(n, vc, knvc, knpvc, v, knv, knpv)
        # upart_off = self.upart_off(n, uc, jnuc, u, jnu)
        # vpart_off = self.vpart_off(n, vc, knvc, v, knv)
        # return np.sqrt(np.real(
        #     val_u * (
        #         a * (ac * upart_diag + bc * upart_off) +
        #         b * (bc * upart_diag + ac * upart_off)) -
        #     val_v * (
        #         a * (ac * vpart_diag + bc * vpart_off) +
        #         b * (bc * vpart_diag + ac * vpart_off))))

    @staticmethod
    def upart_diag(n, uc, jnuc, jnpuc, u, jnu, jnpu):
        if abs(uc - u) < 1e-10:
            u0 = (u + uc) / 2
            jnu0 = jv(n, u0)
            jnpu0 = jvp(n, u0)
            return (
                jnu0 * jnpu0 / u0
                + (jnpu0 ** 2 + (1 - n ** 2 / u0 ** 2) * jnu0 ** 2) / 2
            )
        if abs(uc + u) < 1e-10:
            u0 = (u - uc) / 2
            jnu0 = jv(n, u0)
            jnpu0 = jvp(n, u0)
            return (-1) ** (n - 1) * (
                jnu0 * jnpu0 / u0
                + (jnpu0 ** 2 + (1 - n ** 2 / u0 ** 2) * jnu0 ** 2) / 2
            )
        return (uc * jnuc * jnpu - u * jnu * jnpuc) / (uc ** 2 - u ** 2)

    @staticmethod
    def upart_off(n, uc, jnuc, u, jnu):
        return n * (jnuc * jnu) / (uc * u)

    @staticmethod
    def vpart_diag(n, vc, knvc, knpvc, v, knv, knpv):
        if abs(vc - v) < 1e-10:
            v0 = (v + vc) / 2
            knv0 = kv(n, v0)
            knpv0 = kvp(n, v0)
            return (
                knv0 * knpv0 / v0
                + (knpv0 ** 2 - (1 + n ** 2 / v0 ** 2) * knv0 ** 2) / 2
            )
        if abs(vc + v) < 1e-10:
            v0 = (v - vc) / 2
            knv0 = kv(n, v0)
            knpv0 = kvp(n, v0)
            return (-1) ** (n - 1) * (
                knv0 * knpv0 / v0
                + (knpv0 ** 2 - (1 + n ** 2 / v0 ** 2) * knv0 ** 2) / 2
            )
        return (vc * knvc * knpv - v * knv * knpvc) / (vc ** 2 - v ** 2)

    @staticmethod
    def vpart_off(n, vc, knvc, v, knv):
        return n * (knvc * knv) / (vc * v)

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
            jnu = jv(n, u)
            jnpu = jvp(n, u)
            v = self.samples.v(h ** 2, w, e2)
            knv = kv(n, v)
            knpv = kvp(n, v)
            val_u = 2 * np.pi * self.r ** 2 / en
            val_v = val_u * ((u * jnu) / (v * knv)) ** 2
            upart_diag = self.upart_diag(n, u, jnu, jnpu, u, jnu, jnpu)
            vpart_diag = self.vpart_diag(n, v, knv, knpv, v, knv, knpv)
            upart_off = self.upart_off(n, u, jnu, u, jnu)
            vpart_off = self.vpart_off(n, v, knv, v, knv)
            val = val_u * (
                h / w * a * (a * upart_diag + b * upart_off)
                + e1 * w / h * b * (b * upart_diag + a * upart_off)
            ) - val_v * (
                h / w * a * (a * vpart_diag + b * vpart_off)
                + e2 * w / h * b * (b * vpart_diag + a * vpart_off)
            )
        return val

    def Yab(self, w, h1, s1, l1, n1, m1, a1, b1, h2, s2, l2, n2, m2, a2, b2):
        """Return the admittance matrix element of the waveguide modes using
        orthogonality.

        Args:
            w: A complex indicating the angular frequency
            h1: A complex indicating the phase constant.
            s1: 0 for TE-like mode or 1 for TM-like mode
            l1: 0 for h mode or 1 for v mode
            n1: the order of the mode
            m1: the number of modes in the order and the polarization
            a1: A complex indicating the coefficient of TE-component
            b1: A complex indicating the coefficient of TM-component
            h2: A complex indicating the phase constant.
            s2: 0 for TE-like mode or 1 for TM-like mode
            l2: 0 for h mode or 1 for v mode
            n2: the order of the mode
            m2: the number of modes in the order and the polarization
            a2: A complex indicating the coefficient of TE-component
            b2: A complex indicating the coefficient of TM-component
        Returns:
            y: A complex indicating the effective admittance
        """
        if n1 != n2 or l1 != l2:
            return 0.0
        n = n1
        e1 = self.fill(w)
        e2 = self.clad(w)
        en = 1 if n == 0 else 2
        if e2.real < -1e6:
            if s1 != s2 or m1 != m2:
                return 0.0
            if s1 == 0:
                val = h2 / w
            else:
                val = e1 * w / h2
        else:
            ac = a1
            bc = b1
            a, b = a2, b2
            uc = self.samples.u(h1 ** 2, w, e1)
            vc = self.samples.v(h1 ** 2, w, e2)
            u = self.samples.u(h2 ** 2, w, e1)
            v = self.samples.v(h2 ** 2, w, e2)
            jnuc = jv(n, uc)
            jnpuc = jvp(n, uc)
            knvc = kv(n, vc)
            knpvc = kvp(n, vc)
            jnu = jv(n, u)
            jnpu = jvp(n, u)
            knv = kv(n, v)
            knpv = kvp(n, v)
            val_u = 2 * np.pi * self.r ** 2 / en
            val_v = val_u * (uc * u * jnuc * jnu) / (vc * v * knvc * knv)
            upart_diag = self.upart_diag(n, uc, jnuc, jnpuc, u, jnu, jnpu)
            vpart_diag = self.vpart_diag(n, vc, knvc, knpvc, v, knv, knpv)
            upart_off = self.upart_off(n, uc, jnuc, u, jnu)
            vpart_off = self.vpart_off(n, vc, knvc, v, knv)
            val = val_u * (
                h2 / w * a * (ac * upart_diag + bc * upart_off)
                + e1 * w / h2 * b * (bc * upart_diag + ac * upart_off)
            ) - val_v * (
                h2 / w * a * (ac * vpart_diag + bc * vpart_off)
                + e2 * w / h2 * b * (bc * vpart_diag + ac * vpart_off)
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
            er_te = (jv(n - 1, ur) + jv(n + 1, ur)) / 2 * fr
            er_tm = jvp(n, ur) * fr
            er = a * er_te + b * er_tm
            ep_te = jvp(n, ur) * fp
            ep_tm = (jv(n - 1, ur) + jv(n + 1, ur)) / 2 * fp
            ep = a * ep_te + b * ep_tm
            ez = u / (1j * h * self.r) * b * jv(n, ur) * fr
            hr = -y_te * a * ep_te - y_tm * b * ep_tm
            hp = y_te * a * er_te + y_tm * b * er_tm
            hz = -u / (1j * h * self.r) * y_te * a * jv(n, ur) * fp
        else:
            y_tm = self.y_tm_outer(w, h)
            val = -u * jv(n, u) / (v * kv(n, v))
            er_te = -(kv(n - 1, vr) - kv(n + 1, vr)) / 2 * fr * val
            er_tm = kvp(n, vr) * fr * val
            er = a * er_te + b * er_tm
            ep_te = kvp(n, vr) * fp * val
            ep_tm = -(kv(n - 1, vr) - kv(n + 1, vr)) / 2 * fp * val
            ep = a * ep_te + b * ep_tm
            ez = -v / (1j * h * self.r) * b * kv(n, vr) * fr * val
            hr = -y_te * a * ep_te - y_tm * b * ep_tm
            hp = y_te * a * er_te + y_tm * b * er_tm
            hz = v / (1j * h * self.r) * y_te * a * kv(n, vr) * fp * val
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
            er_te = (jv(n - 1, ur) + jv(n + 1, ur)) / 2 * fr
            er_tm = jvp(n, ur) * fr
            er = a * er_te + b * er_tm
            ep_te = jvp(n, ur) * fp
            ep_tm = (jv(n - 1, ur) + jv(n + 1, ur)) / 2 * fp
            ep = a * ep_te + b * ep_tm
            ez = u / (1j * h * self.r) * b * jv(n, ur) * fr
        else:
            val = -u * jv(n, u) / (v * kv(n, v))
            er_te = -(kv(n - 1, vr) - kv(n + 1, vr)) / 2 * fr * val
            er_tm = kvp(n, vr) * fr * val
            er = a * er_te + b * er_tm
            ep_te = kvp(n, vr) * fp * val
            ep_tm = -(kv(n - 1, vr) - kv(n + 1, vr)) / 2 * fp * val
            ep = a * ep_te + b * ep_tm
            ez = -v / (1j * h * self.r) * b * kv(n, vr) * fr * val
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
            er_te = (jv(n - 1, ur) + jv(n + 1, ur)) / 2 * fr
            er_tm = jvp(n, ur) * fr
            ep_te = jvp(n, ur) * fp
            ep_tm = (jv(n - 1, ur) + jv(n + 1, ur)) / 2 * fp
            hr = -y_te * a * ep_te - y_tm * b * ep_tm
            hp = y_te * a * er_te + y_tm * b * er_tm
            hz = -u / (1j * h * self.r) * y_te * a * jv(n, ur) * fp
        else:
            y_tm = self.y_tm_outer(w, h)
            val = -u * jv(n, u) / (v * kv(n, v))
            er_te = -(kv(n - 1, vr) - kv(n + 1, vr)) / 2 * fr * val
            er_tm = kvp(n, vr) * fr * val
            ep_te = kvp(n, vr) * fp * val
            ep_tm = -(kv(n - 1, vr) - kv(n + 1, vr)) / 2 * fp * val
            hr = -y_te * a * ep_te - y_tm * b * ep_tm
            hp = y_te * a * er_te + y_tm * b * er_tm
            hz = v / (1j * h * self.r) * y_te * a * kv(n, vr) * fp * val
        hx = hr * np.cos(p) - hp * np.sin(p)
        hy = hr * np.sin(p) + hp * np.cos(p)
        return np.array([hx, hy, hz])

    @staticmethod
    def u_jnu_jnpu_pec(num_n, num_m):
        us = np.empty((2, num_n, num_m))
        jnus = np.empty((2, num_n, num_m))
        jnpus = np.empty((2, num_n, num_m))
        for n in range(num_n):
            us[0, n] = jnp_zeros(n, num_m)
            us[1, n] = jn_zeros(n, num_m)
            jnus[0, n] = jv(n, us[0, n])
            jnus[1, n] = np.zeros(num_m)
            jnpus[0, n] = np.zeros(num_m)
            jnpus[1, n] = jvp(n, us[1, n])
        return us, jnus, jnpus
