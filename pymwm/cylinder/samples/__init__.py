# -*- coding: utf-8 -*-
from typing import Dict, Tuple, List
import numpy as np
from scipy.special import jv, jvp, kv, kvp, jn_zeros, jnp_zeros
from pyoptmat import Material
from pymwm.waveguide import Sampling


class Samples(Sampling):
    """A class defining samples of phase constants of cylindrical waveguide
    modes.

    Attributes:
        r: A float indicating the radius of the circular cross section [um].
    """

    def __init__(self, size: float, fill: Material, clad: Material,
                 params: Dict):
        """Init Samples class.

        Args:
            size: A float indicating the radius of the cross section [um]
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
        super().__init__(size, fill, clad, params)
        self.r = size

    @property
    def shape(self):
        return 'cylinder'

    @property
    def num_all(self):
        return self.params['num_n'] * (2 * self.params['num_m'] + 1)

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
        num_m = self.params['num_m']
        chi = jn_zeros(n, num_m + 1)
        h2s_mag = self.fill(w_comp) * w_comp ** 2 - chi ** 2 / self.r ** 2
        chi = jnp_zeros(n, num_m)
        h2s_elec = self.fill(w_comp) * w_comp ** 2 - chi ** 2 / self.r ** 2
        h2s = np.hstack((h2s_mag, h2s_elec))
        return h2s

    def beta2_pec_per_mode(self, w, key):
        """Return squares of phase constants for a PEC waveguide

        Args:
            w: A complex indicating the angular frequency
            key: A tuple (pol, n, m) where
                pol: 'E' or 'M' indicating the polarization.
                n: A integer indicating the order of the mode.
                m: A integer indicating the ordinal of the mode in the same
                    order.
        Returns:
            h2s: A 1D array indicating squares of phase constants, whose first
                num_m+1 elements are for TM-like modes and the rest are for
                TE-like modes.
        """
        w_comp = w.real + 1j * w.imag
        pol, n, m = key
        if pol == 'M':
            chi = jn_zeros(n, m)[-1]
            h2 = self.fill(w_comp) * w_comp ** 2 - chi ** 2 / self.r ** 2
        else:
            chi = jnp_zeros(n, m)[-1]
            h2 = self.fill(w_comp) * w_comp ** 2 - chi ** 2 / self.r ** 2
        return h2

    def u(self, h2: complex, w: complex, e1: complex) -> complex:
        # return np.sqrt(e1 * w ** 2 - h2) * self.r
        return (1 + 1j) * np.sqrt(-0.5j * (e1 * w ** 2 - h2)) * self.r

    def v(self, h2: complex, w: complex, e2: complex) -> complex:
        # This definition is very important!!
        # Other definitions can not give good results in some cases
        return (1 - 1j) * np.sqrt(0.5j * (- e2 * w ** 2 + h2)) * self.r

    def eig_eq(self, h2: complex, w: complex, pol: str, n: int,
               e1: complex, e2: complex):
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
        jus = jv(n, u)
        jpus = jvp(n, u)
        kvs = kv(n, v)
        kpvs = kvp(n, v)
        te = jpus / u + kpvs * jus / (v * kvs)
        tm = e1 * jpus / u + e2 * kpvs * jus / (v * kvs)
        if n == 0:
            if pol == 'M':
                val = tm
            else:
                val = te
        else:
            val = (tm * te - h2comp * (n / w) ** 2 *
                   ((1 / u ** 2 + 1 / v ** 2) * jus) ** 2)
        return val

    def jac(self, h2, args):
        """Return Jacobian of the characteristic equation

        Args:
            h2: A complex indicating the square of the phase constant.
            args: A tuple (w, n, e1, e2), where w indicates the angular
                frequency, n indicates  the order of the modes, e1 indicates
                the permittivity of the core, and e2 indicates the permittivity
                of the clad.
        Returns:
            val: A complex indicating the Jacobian of the characteristic
                equation.
        """
        w, pol, n, e1, e2 = args
        h2comp = h2.real + 1j * h2.imag
        u = self.u(h2comp, w, e1)
        v = self.v(h2comp, w, e2)
        jus = jv(n, u)
        jpus = jvp(n, u)
        kvs = kv(n, v)
        kpvs = kvp(n, v)
        du_dh2 = - self.r / (2 * u)
        dv_dh2 = self.r / (2 * v)
        te = jpus / u + kpvs * jus / (v * kvs)
        dte_du = (-(u * (1 - n ** 2 / u ** 2) * jus + 2 * jpus) / u ** 2 +
                  jpus * kpvs / (v * kvs))
        dte_dv = jus * (n ** 2 * kvs ** 2 / v + v * (kvs ** 2 - kpvs ** 2) -
                        2 * kvs * kpvs) / (v ** 2 * kvs ** 2)
        tm = e1 * jpus / u + e2 * kpvs * jus / (v * kvs)
        dtm_du = e1 * dte_du
        dtm_dv = e2 * dte_dv
        if n == 0:
            if pol == 'M':
                val = dtm_du * du_dh2 + dtm_dv * dv_dh2
            else:
                val = dte_du * du_dh2 + dte_dv * dv_dh2
        else:
            dre_dh2 = -(n / w) ** 2 * jus * (
                jus * (
                    (1 / u ** 2 + 1 / v ** 2) ** 2 -
                    self.r * h2comp * (1 / u ** 4 - 1 / v ** 4)) +
                jpus * 2 * h2comp * (1 / u ** 2 + 1 / v ** 2) ** 2)
            val = ((dte_du * du_dh2 + dte_dv * dv_dh2) * tm +
                   (dtm_du * du_dh2 + dtm_dv * dv_dh2) * te +
                   dre_dh2)
        return val

    def func_jac(self, h2, *args):
        """Return the value and Jacobian of the characteristic equation

        Args:
            h2: A complex indicating the square of the phase constant.
        Returns:
            val: 2 complexes indicating the left-hand value and Jacobian
                of the characteristic equation.
        """
        w, pol, n, e1, e2 = args
        h2comp = h2.real + 1j * h2.imag
        u = self.u(h2comp, w, e1)
        v = self.v(h2comp, w, e2)
        jus = jv(n, u)
        jpus = jvp(n, u)
        kvs = kv(n, v)
        kpvs = kvp(n, v)
        du_dh2 = - self.r / (2 * u)
        dv_dh2 = self.r / (2 * v)
        te = jpus / u + kpvs * jus / (v * kvs)
        dte_du = (-(u * (1 - n ** 2 / u ** 2) * jus + 2 * jpus) / u ** 2 +
                  jpus * kpvs / (v * kvs))
        dte_dv = jus * (n ** 2 * kvs ** 2 / v + v * (kvs ** 2 - kpvs ** 2) -
                        2 * kvs * kpvs) / (v ** 2 * kvs ** 2)
        tm = e1 * jpus / u + e2 * kpvs * jus / (v * kvs)
        dtm_du = e1 * dte_du
        dtm_dv = e2 * dte_dv
        if n == 0:
            if pol == 'M':
                f = tm
                val = dtm_du * du_dh2 + dtm_dv * dv_dh2
            else:
                f = te
                val = dte_du * du_dh2 + dte_dv * dv_dh2
        else:
            f = (tm * te - h2comp * (n / w) ** 2 *
                 ((1 / u ** 2 + 1 / v ** 2) * jus) ** 2)
            dre_dh2 = -(n / w) ** 2 * jus * (
                jus * (
                    (1 / u ** 2 + 1 / v ** 2) ** 2 -
                    self.r * h2comp * (1 / u ** 4 - 1 / v ** 4)) +
                jpus * 2 * h2comp * (1 / u ** 2 + 1 / v ** 2) ** 2)
            val = ((dte_du * du_dh2 + dte_dv * dv_dh2) * tm +
                   (dtm_du * du_dh2 + dtm_dv * dv_dh2) * te +
                   dre_dh2)
        return f, val

    def beta2(self, w, n, e1, e2, xis):
        """Return roots and convergences of the characteristic equation

        Args:
            w: A complex indicating the angular frequency.
            n: A integer indicating the order of the mode
            e1: A complex indicating the permittivity of tha core.
            e2: A complex indicating the permittivity of tha clad.
            xis: A complex indicating the initial approximations for the roots
                whose number of elements is 2*num_m+1.
        Returns:
            xs: A 1D array indicating the roots, whose length is 2*num_m+1.
            success: A 1D array indicating the convergence information for xs.
        """
        if self.clad.model == 'pec':
            xs = self.beta2_pec(w, n)
            success = np.ones_like(xs, dtype=bool)
            return xs, success
        from scipy.optimize import root
        num_m = self.params['num_m']
        roots = []
        vals = []
        success = []

        # def func(h2vec, *args):
        #     h2 = h2vec[0] + h2vec[1] * 1j
        #     f, ja = self.func_jac(h2, *args)
        #     prod_denom = 1.0
        #     sum_denom = 0.0
        #     for h2_0 in roots:
        #         denom = h2 - h2_0
        #         while (abs(denom) < 1e-9):
        #             denom += 1.0e-9
        #         prod_denom *= 1.0 / denom
        #         sum_denom += 1.0 / denom
        #     f *= prod_denom
        #     ja = ja * prod_denom - f * sum_denom
        #     f_array = np.array([f.real, f.imag])
        #     ja_array = np.array([[ja.real, ja.imag],
        #                          [ja.imag, ja.real]])
        #     return f_array, ja_array

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
            if i < num_m + 1:
                args = (w, 'M', n, e1, e2)
            else:
                args = (w, 'E', n, e1, e2)
            # result = root(func, (xi.real, xi.imag), args=args, jac=True,
            #               method='hybr', options={'xtol': 1.0e-9})
            result = root(func, np.array([xi.real, xi.imag]), args=args,
                          jac=False, method='hybr', options={'xtol': 1.0e-9})
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
        if self.clad.model == 'pec':
            xs = self.beta2_pec(self.ws[0], n)
            success = np.ones_like(xs, dtype=bool)
            return xs, success
        w_0 = 0.1
        e1 = self.fill(w_0)
        e2_0 = -1.0e7 + self.clad(w_0).imag * 1j
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

    def beta2_w_max(self, n):
        """Return roots and convergences of the characteristic equation at
            the highest angular frequency, ws[-1].

        Args:
            n: A integer indicating the order of the mode
        Returns:
            xs: A 1D array indicating the roots, whose length is 2*num_m+1.
            success: A 1D array indicating the convergence information for xs.
        """
        w = self.ws[-1]
        xis = xs = self.beta2_pec(w, n)
        success = np.ones_like(xs, dtype=bool)
        if self.clad.model == 'pec':
            return xs, success
        e1 = self.fill(w)
        e2_0 = -1.0e7 + self.clad(w).imag * 1j
        de2 = (self.clad(w) - e2_0) / 100000
        for i in range(100001):
            e2 = e2_0 + de2 * i
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

    def _betas_convs(self, n, xs_array, success_array):
        num_m = self.params['num_m']
        betas = {}
        convs = {}
        for m in range(1, num_m + 2):
            betas[('M', n, m)] = np.zeros((len(self.ws), len(self.wis)),
                                          dtype=complex)
            convs[('M', n, m)] = np.zeros((len(self.ws), len(self.wis)),
                                          dtype=bool)
        for m in range(1, num_m + 1):
            betas[('E', n, m)] = np.zeros((len(self.ws), len(self.wis)),
                                          dtype=complex)
            convs[('E', n, m)] = np.zeros((len(self.ws), len(self.wis)),
                                          dtype=bool)
        for iwi in range(len(self.wis)):
            for iwr in range(len(self.ws)):
                for i in range(num_m + 1):
                    betas[('M', n, i + 1)][iwr, iwi] = self.beta_from_beta2(
                        xs_array[iwr, iwi][i])
                    convs[('M', n, i + 1)][iwr, iwi] = success_array[
                        iwr, iwi][i]
                for i in range(num_m):
                    betas[('E', n, i + 1)][iwr, iwi] = self.beta_from_beta2(
                        xs_array[iwr, iwi][i + num_m + 1])
                    convs[('E', n, i + 1)][iwr, iwi] = success_array[
                        iwr, iwi][i + num_m + 1]
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
        num_m = self.params['num_m']
        xs_array = np.zeros((len(self.ws), len(self.wis),
                             2 * num_m + 1), dtype=complex)
        success_array = np.zeros((len(self.ws), len(self.wis),
                                  2 * num_m + 1), dtype=bool)
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
                    xis = (xs_array[iwr, iwi - 1] +
                           xs_array[iwr - 1, iwi] -
                           xs_array[iwr - 1, iwi - 1])
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
        num_n = self.params['num_n']
        num_m = self.params['num_m']
        betas = {}
        convs = {}
        for n in range(num_n):
            xs_array, success_array = xs_success_list[n]
            for m in range(1, num_m + 2):
                betas[('M', n, m)] = np.zeros((len(self.ws), len(self.wis)),
                                              dtype=complex)
                convs[('M', n, m)] = np.zeros((len(self.ws), len(self.wis)),
                                              dtype=bool)
            for m in range(1, num_m + 1):
                betas[('E', n, m)] = np.zeros((len(self.ws), len(self.wis)),
                                              dtype=complex)
                convs[('E', n, m)] = np.zeros((len(self.ws), len(self.wis)),
                                              dtype=bool)
            for iwi in range(len(self.wis)):
                for iwr in range(len(self.ws)):
                    w = self.ws[iwr] + 1j * self.wis[iwi]
                    e2 = self.clad(w)
                    for i in range(num_m + 1):
                        x = xs_array[iwr, iwi][i]
                        v = self.v(x, w, e2)
                        betas[
                            ('M', n, i + 1)][iwr, iwi] = self.beta_from_beta2(
                                x)
                        convs[('M', n, i + 1)][iwr, iwi] = (
                            success_array[iwr, iwi][i]
                            if v.real > abs(v.imag) else False)
                    for i in range(num_m):
                        x = xs_array[iwr, iwi][i + num_m + 1]
                        v = self.v(x, w, e2)
                        betas[
                            ('E', n, i + 1)][iwr, iwi] = self.beta_from_beta2(
                                x)
                        convs[('E', n, i + 1)][iwr, iwi] = (
                            success_array[iwr, iwi][i + num_m + 1]
                            if v.real > abs(v.imag) else False)
        return betas, convs


class SamplesLowLoss(Samples):
    """A class defining samples of phase constants of cylindrical waveguide
    modes in a virtual low-loss clad waveguide by subclassing the Samples
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

    def __init__(self, r, fill, clad, params):
        """Init Samples class.

        Args:
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
        """
        super(SamplesLowLoss, self).__init__(r, fill, clad, params)

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
        num_n = self.params['num_n']
        num_m = self.params['num_m']
        betas = {}
        convs = {}
        for n in range(num_n):
            for m in range(1, num_m + 2):
                betas[('M', n, m)] = np.zeros((len(self.ws), len(self.wis)),
                                              dtype=complex)
                convs[('M', n, m)] = np.zeros((len(self.ws), len(self.wis)),
                                              dtype=bool)
            for m in range(1, num_m + 1):
                betas[('E', n, m)] = np.zeros((len(self.ws), len(self.wis)),
                                              dtype=complex)
                convs[('E', n, m)] = np.zeros((len(self.ws), len(self.wis)),
                                              dtype=bool)
        for iwr in range(num_iwr):
            for iwi in range(num_iwi):
                j = iwr * num_iwi + iwi
                w = self.ws[iwr] + 1j * self.wis[iwi]
                e2 = self.clad(w)
                for n in range(num_n):
                    for i in range(num_m + 1):
                        x = xs_success_list[j][0][n][i]
                        v = self.v(x, w, e2)
                        betas[('M', n, i + 1)][iwr, iwi] = (
                            self.beta_from_beta2(x))
                        convs[('M', n, i + 1)][iwr, iwi] = (
                            xs_success_list[j][1][n][i]
                            if v.real > abs(v.imag) else False)
                    for i in range(num_m):
                        x = xs_success_list[j][0][n][i + num_m + 1]
                        v = self.v(x, w, e2)
                        betas[('E', n, i + 1)][iwr, iwi] = (
                            self.beta_from_beta2(x))
                        convs[('E', n, i + 1)][iwr, iwi] = (
                            xs_success_list[j][1][n][i + num_m + 1]
                            if v.real > abs(v.imag) else False)
        return betas, convs
