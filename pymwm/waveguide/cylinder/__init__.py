# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import jv, jvp, kv, kvp, jn_zeros, jnp_zeros


class Cylinder(object):
    """A class defining a cylindrical waveguide.

     Attributes:
        fill: An object of Material class for the core
        clad: An object of Material class for the clad
        r: A float indicating the radius of the circular cross section [um].
        num_n: An integer indicating the number of orders of modes.
        num_m: An integer indicating the number of modes in each order
            and polarization.
        ws: A 1D array indicating the real part of the angular frequencies
            to be calculated [rad * (c / 1um)] =[2.99792458e14 rad / s].
        wis: A 1D array indicating the imaginary part of the angular
            frequencies to be calculated [rad * (c / 1um)].
    """

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
                'modes': A dict of the settings for calculating modes:
                    'lmax': A float indicating the maximum wavelength [um]
                    'lmin': A float indicating the minimum wavelength [um]
                    'limag': A float indicating the minimum value of
                        (c / fimag) [um] where fimag is the imaginary part of
                        the frequency.
                    'nwr': A integer indicating the number of points taken
                        in the real axis of angular frequency.
                    'nwi': A integer indicating the number of points taken
                        in the imaginary axis of angular frequency.
                num_n: An integer indicating the number of orders of modes.
                num_m: An integer indicating the number of modes in each order
                        and polarization.
        """
        from pymwm.material import Material
        self.r = params['core']['size']
        p_fill = params['core']['fill']
        p_clad = params['clad']
        self.fill = Material(p_fill)
        self.clad = Material(p_clad)
        p = params['modes']
        self.num_n = p['num_n']
        self.num_m = p['num_m']
        self.alpha_list = []
        pol = 'M'
        for n in range(self.num_n):
            for m in range(1, self.num_m + 1):
                self.alpha_list.append((pol, n, m))
        pol = 'E'
        for n in range(1, self.num_n):
            for m in range(1, self.num_m):
                self.alpha_list.append((pol, n, m))
        wmin = 2 * np.pi / p['lmax']
        wmax = 2 * np.pi / p['lmin']
        wimag = 2 * np.pi / p['limag']
        self.ws = np.linspace(wmin, wmax, p['nwr'] + 1)
        self.wis = np.linspace(0.0, -wimag, p['nwi'] + 1)

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
        wcomp = w.real + 1j * w.imag
        # The number of TM-like modes for each order is taken lager than
        # the number of TE-like modes since TM-like modes are almost identical
        # to those for PEC waveguide and can be calculated easily.
        # Dividing function by (x - x0) where x0 is already-found root
        # makes it easier to find new roots.
        chi = jn_zeros(n, self.num_m + 1)
        h2sM = self.fill(wcomp) * wcomp ** 2 - chi ** 2 / self.r ** 2
        chi = jnp_zeros(n, self.num_m)
        h2sE = self.fill(wcomp) * wcomp ** 2 - chi ** 2 / self.r ** 2
        h2s = np.hstack((h2sM, h2sE))
        return h2s

    def u(self, h2, w, e1):
        return np.sqrt(e1 * w ** 2 - h2) * self.r

    def v(self, h2, w, e2):
        return np.sqrt(- e2 * w ** 2 + h2) * self.r
        # return -1j * np.sqrt(e2 * w ** 2 - h2) * self.r

    def eigeq(self, h2, args):
        """Return the value of the characteristic equation

        Args:
            h2: A complex indicating the square of the phase constant.
            args: A tuple (w, n, e1, e2), where w indicates the angular
                frequency, n indicates  the order of the modes, e1 indicates
                the permittivity of the core, and e2 indicates the permittivity
                of the clad.
        Returns:
            hs: A 1D array indicating phase constants, whose length
                is 2*num_m+1.
        """
        w, n, e1, e2 = args
        x = h2.real + 1j * h2.imag
        u = self.u(x, w, e1)
        v = self.v(x, w, e2)
        jus = jv(n, u)
        jpus = jvp(n, u)
        kvs = kv(n, v)
        kpvs = kvp(n, v)
        te = jpus / u + kpvs * jus / (v * kvs)
        tm = e1 * jpus / u + e2 * kpvs * jus / (v * kvs)
        return (tm * te - x * (n / w) ** 2 *
                ((1 / u ** 2 + 1 / v ** 2) * jus) ** 2)

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
        from scipy.optimize import root
        roots = []
        vals = []
        success = []
        for xi in xis:

            def func(h2vec):
                h2 = h2vec[0] + h2vec[1] * 1j
                val = self.eigeq(h2, (w, n, e1, e2))
                for h2_0 in roots:
                    denom = h2 - h2_0
                    # while (abs(denom) < 1e-8):
                    #     denom += 1.0e-8
                    val /= denom
                return np.array([val.real, val.imag])

            result = root(func, (xi.real, xi.imag), method='hybr',
                          options={'xtol': 1.0e-10})
            x = result.x[0] + result.x[1] * 1j
            if self.v(x, w, e2).imag < 0:
                if result.success:
                    roots.append(x)
                success.append(result.success)
            else:
                success.append(False)
            vals.append(x)
        return np.array(vals), success

    def beta2_wmin(self, n):
        """Return roots and convergences of the characteristic equation at
            the lowest angular frequency, ws[0].

        Args:
            n: A integer indicating the order of the mode
        Returns:
            xs: A 1D array indicating the roots, whose length is 2*num_m+1.
            success: A 1D array indicating the convergence information for xs.
        """
        w_0 = 0.1
        xs = self.beta2_pec(w_0, n)
        e1 = self.fill(w_0)
        e2_0 = 1.0e7j
        de2 = (self.clad(w_0) - e2_0) / 1000
        for i in range(1001):
            e2 = e2_0 + de2 * i
            xs, success = self.beta2(w_0, n, e1, e2, xs)
            # print(i, e2)
            # print(xs)
        dw = (self.ws[0] - w_0) / 1000
        for i in range(1001):
            w = w_0 + dw * i
            e1 = self.fill(w)
            e2 = self.clad(w)
            xs, success = self.beta2(w, n, e1, e2, xs)
            # print(i, w, e2)
            # print(xs)
        return xs, success

    def beta_from_beta2(self, x):
        val = np.sqrt(x)
        if ((abs(val.real) > abs(val.imag) and val.real < 0) or
           (abs(val.real) < abs(val.imag) and val.imag < 0)):
            val *= -1
        return val

    def __call__(self, n):
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
        xs_array = np.zeros((len(self.ws), 2 * self.num_m + 1), dtype=complex)
        betas = {}
        convs = {}
        iwr = iwi = 0
        wr = self.ws[iwr]
        wi = self.wis[iwi]
        w = wr + 1j * wi
        e1 = self.fill(w)
        e2 = self.clad(w)
        xs, success = self.beta2_wmin(n)
        xs_array[iwr] = xs
        for pol in ['M', 'E']:
            for m in range(1, self.num_m + 1):
                betas[(pol, n, m)] = np.zeros((len(self.ws), len(self.wis)),
                                              dtype=complex)
                convs[(pol, n, m)] = np.zeros((len(self.ws), len(self.wis)),
                                              dtype=bool)
        for i in range(self.num_m):
            betas[('M', n, i + 1)][iwr, iwi] = self.beta_from_beta2(xs[i])
            convs[('M', n, i + 1)][iwr, iwi] = success[i]
            if n != 0:
                betas[('E', n, i + 1)][iwr, iwi] = self.beta_from_beta2(
                    xs[i + self.num_m + 1])
                convs[('E', n, i + 1)][iwr, iwi] = success[
                    i + self.num_m + 1]
        for iwr in range(1, len(self.ws)):
            wr = self.ws[iwr]
            wi = self.wis[iwi]
            w = wr + 1j * wi
            e1 = self.fill(w)
            e2 = self.clad(w)
            xs, success = self.beta2(w, n, e1, e2, xs)
            xs_array[iwr] = xs
            for i in range(self.num_m):
                betas[('M', n, i + 1)][iwr, iwi] = self.beta_from_beta2(xs[i])
                convs[('M', n, i + 1)][iwr, iwi] = success[i]
                if n != 0:
                    betas[('E', n, i + 1)][iwr, iwi] = self.beta_from_beta2(
                        xs[i + self.num_m + 1])
                    convs[('E', n, i + 1)][iwr, iwi] = success[
                        i + self.num_m + 1]
        for iwi in range(1, len(self.wis)):
            for iwr in range(len(self.ws)):
                wr = self.ws[iwr]
                wi = self.wis[iwi]
                w = wr + 1j * wi
                e1 = self.fill(w)
                e2 = self.clad(w)
                xs = xs_array[iwr]
                xs, success = self.beta2(w, n, e1, e2, xs)
                xs_array[iwr] = xs
                for i in range(self.num_m):
                    betas[('M', n, i + 1)][iwr, iwi] = self.beta_from_beta2(
                        xs[i])
                    convs[('M', n, i + 1)][iwr, iwi] = success[i]
                    if n != 0:
                        betas[('E', n, i + 1)][iwr, iwi] = (
                            self.beta_from_beta2(xs[i + self.num_m + 1]))
                        convs[('E', n, i + 1)][iwr, iwi] = success[
                            i + self.num_m + 1]
        return betas, convs
