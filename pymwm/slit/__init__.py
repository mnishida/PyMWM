# -*- coding: utf-8 -*-
import numpy as np
from pymwm.slit.utils import coefs_cython, ABY_cython


class Slit(object):
    """A class defining a slit waveguide.

     Attributes:
        fill: An instance of Material class for the core
        clad: An instance of Material class for the clad
        r: A float indicating the width of the slit [um].
        samples: An instance of Samples class
    """

    def __init__(self, params):
        """Init Slit class.

        Args:
            params: A dict whose keys and values are as follows:
                'core': A dict of the setting parameters of the core:
                    'shape': A string indicating the shape of the core.
                    'size': A float indicating the width of the slit [um].
                    'fill': A dict of the parameters of the core Material.
                'clad': A dict of the parameters of the clad Material.
                'bounds': A dict indicating the bounds of interpolation, and
                    its keys and values are as follows:
                    'lmax': A float indicating the maximum wavelength [um]
                    'lmin': A float indicating the minimum wavelength [um]
                    'limag': A float indicating the minimum value of
                        abs(c / fimag) [um] where fimag is the imaginary part
                        of the frequency.
                'modes': A dict of the settings for calculating modes:
                    'lmax': A float indicating the maximum wavelength [um]
                        (defulat: 5.0)
                    'lmin': A float indicating the minimum wavelength [um]
                        (defulat: 0.4)
                    'limag': A float indicating the minimum value of
                        abs(c / fimag) [um] where fimag is the imaginary part
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
        from pyoptmat import Material
        from pymwm.slit.samples import Samples
        self.r = params['core']['size']
        self.fill = Material(params['core']['fill'])
        self.clad = Material(params['clad'])
        im_factor = 1.0
        if self.clad.im_factor != 1.0:
            im_factor = self.clad.im_factor
            self.clad.im_factor = 1.0
        pmodes = params['modes'].copy()
        num_n_0 = pmodes['num_n']
        success = False
        for num_n in [n for n in range(num_n_0, 25)]:
            pmodes['num_n'] = num_n
            self.samples = Samples(
                self.r, self.fill, self.clad, pmodes)
            try:
                betas, convs = self.samples.load()
                success = True
                break
            except:
                continue
        if not success:
            pmodes['num_n'] = num_n_0
            self.samples = Samples(
                self.r, self.fill, self.clad, pmodes)
            from multiprocessing import Pool
            num_n = params['modes']['num_n']
            p = Pool(num_n)
            betas_list = p.map(self.samples, range(num_n))
            betas = {key: val for betas, convs in betas_list
                     for key, val in betas.items()}
            convs = {key: val for betas, convs in betas_list
                     for key, val in convs.items()}
            self.samples.save(betas, convs)
        if im_factor != 1.0:
            self.clad.im_factor = im_factor
            pmodes['num_n'] = num_n_0
            self.samples = Samples(
                self.r, self.fill, self.clad, pmodes)
            success = False
            for num_n in [n for n in range(num_n_0, 25)]:
                pmodes['num_n'] = num_n
                self.samples = Samples(
                    self.r, self.fill, self.clad, pmodes)
                try:
                    betas, convs = self.samples.load()
                    success = True
                    break
                except:
                    continue
            if not success:
                self.clad.im_factor = im_factor
                pmodes['num_n'] = num_n_0
                self.samples = Samples(
                    self.r, self.fill, self.clad, pmodes)
                from multiprocessing import Pool
                num_n = params['modes']['num_n']
                p = Pool(num_n)
                betas_list = p.map(self.samples, range(num_n))
                betas = {key: val for betas, convs in betas_list
                         for key, val in betas.items()}
                convs = {key: val for betas, convs in betas_list
                         for key, val in convs.items()}
                self.samples.save(betas, convs)
        self.bounds = params['bounds']
        self.beta_funcs = self.samples.interpolation(betas, convs, self.bounds)
        self.alpha_list = []
        alpha_M_list = []
        alpha_E_list = []
        for alpha, comp in self.beta_funcs.keys():
            if comp == 'real':
                if alpha[0] == 'M':
                    alpha_M_list.append(alpha)
                else:
                    alpha_E_list.append(alpha)
        alpha_M_list.sort()
        alpha_E_list.sort()
        self.alpha_list = alpha_M_list + alpha_E_list
        self.labels = {}
        for alpha in self.alpha_list:
            pol, n, m = alpha
            if pol == 'E':
                self.labels[alpha] = (
                    r'TE$_{' + r'{0}'.format(n) + r'}$')
            else:
                self.labels[alpha] = (
                    r'TM$_{' + r'{0}'.format(n) + r'}$')
        self.num_n = num_n_0
        self.alphas = {'h': [], 'v': []}
        for alpha in [('E', n, 1) for n in range(1, self.num_n)]:
            if alpha in self.alpha_list:
                self.alphas['v'].append(alpha)
        for alpha in [('M', n, 1) for n in range(self.num_n)]:
            if alpha in self.alpha_list:
                self.alphas['h'].append(alpha)
        self.ls = pmodes.get('ls', ['h', 'v'])
        self.alpha_all = [alpha for l in self.ls for alpha in self.alphas[l]]
        self.l_all = np.array(
                [0 if l == 'h' else 1
                 for l in self.ls for alpha in self.alphas[l]])
        self.s_all = np.array(
                [0 if pol == 'E' else 1
                 for l in self.ls for pol, n, m in self.alphas[l]])
        self.n_all = np.array(
                [n for l in self.ls for pol, n, m in self.alphas[l]])
        self.m_all = np.array(
                [m for l in self.ls for pol, n, m in self.alphas[l]])
        self.num_n_all = self.n_all.shape[0]

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
        hr = self.beta_funcs[(alpha, 'real')](wr, wi)[0, 0]
        hi = self.beta_funcs[(alpha, 'imag')](wr, wi)[0, 0]
        if hr < 0:
            hr = 1e-16
        if hi < 0:
            hi = 1e-16
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
        wcomp = w.real + 1j * w.imag
        pol, n, m = alpha
        val = np.sqrt(self.fill(wcomp) * wcomp ** 2 -
                      (n * np.pi / self.r) ** 2)
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
        if pol == 'E':
            norm = self.norm(w, h, alpha, 1.0 + 0.0j, 0.0j)
            ai, bi = 1.0 / norm, 0.0
        else:
            norm = self.norm(w, h, alpha, 0.0j, 1.0 + 0.0j)
            ai, bi = 0.0, 1.0 / norm
        return ai, bi

    def sinc(self, x):
        x1 = x / np.pi
        return np.sinc(x1)

    def norm(self, w, h, alpha, a, b):
        a2_b2 = np.abs(a) ** 2 + np.abs(b) ** 2
        e1 = self.fill(w)
        e2 = self.clad(w)
        pol, n, m = alpha
        if self.clad(w).real < -1e6:
            if pol == 'M' and n == 0:
                return np.sqrt(a2_b2 * self.r)
            else:
                return np.sqrt(a2_b2 * self.r / 2)
        u = self.samples.u(h ** 2, w, e1)
        uc = u.conjugate()
        v = self.samples.v(h ** 2, w, e2)
        vc = v.conjugate()
        if n % 2 == 0:
            if pol == 'E':
                B_A = np.sin(u)
                parity = -1
            else:
                B_A = u / v * np.sin(u)
                parity = 1
        else:
            if pol == 'E':
                B_A = np.cos(u)
                parity = 1
            else:
                B_A = - u / v * np.cos(u)
                parity = -1
        val = np.sqrt(np.real(a2_b2 * self.r * (
            np.abs(B_A) ** 2 / (v + vc) +
            (self.sinc(u - uc) + parity * self.sinc(u + uc)) / 2)))
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
            if pol == 'E':
                return y_te
            else:
                return y_tm_in
        u = self.samples.u(h ** 2, w, e1)
        v = self.samples.v(h ** 2, w, e2)
        if pol == 'E':
            y_in = y_out = y_te
        else:
            y_in = y_tm_in
            y_out = y_tm_out
        if n % 2 == 0:
            if pol == 'E':
                B_A = np.sin(u)
                parity = -1
            else:
                B_A = u / v * np.sin(u)
                parity = 1
        else:
            if pol == 'E':
                B_A = np.cos(u)
                parity = 1
            else:
                B_A = - u / v * np.cos(u)
                parity = -1
        val = (a ** 2 + b ** 2) * self.r * (
            y_out * B_A ** 2 / (2 * v) +
            (1.0 + parity * self.sinc(2 * u)) * y_in / 2)
        return val

    def Yab(self, w, h1, s1, l1, n1, m1, a1, b1,
            h2, s2, l2, n2, m2, a2, b2):
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
                B_Ac = np.sin(uc)
                B_A = np.sin(u)
                parity = -1
            else:
                B_Ac = uc / vc * np.sin(uc)
                B_A = u / v * np.sin(u)
                parity = 1
        else:
            if s1 == 0:
                B_Ac = np.cos(uc)
                B_A = np.cos(u)
                parity = 1
            else:
                B_Ac = - uc / vc * np.cos(uc)
                B_A = - u / v * np.cos(u)
                parity = -1
        val *= (y_out * B_Ac * B_A / (v + vc) +
                y_in * (self.sinc(u - uc) +
                        parity * self.sinc(u + uc)) / 2)
        return val

    def y_te(self, w, h):
        return h / w

    def y_tm_inner(self, w, h):
        e = self.fill(w)
        return e * w / h

    def y_tm_outer(self, w, h):
        e = self.clad(w)
        return e * w / h

    def fields(self, x, y, w, l, alpha, h, a, b):
        """Return the electromagnetic field vectors for the specified mode and
        point

        Args:
            x: A float indicating the x coordinate [um]
            y: A float indicating the y coordinate [um]
            w: A complex indicating the angular frequency
            l: "h" (horizontal polarization) or "v" (vertical polarization).
                In the slit case, l='h' for TM and l='v' for TE.
            alpha: A tuple (pol, n, m) where pol is 'M' for TM-like mode or
                'E' for TE-like mode, n is the order of the mode, and m is
                the number of modes in the order and the polarization,
                which is always 1 in the slit case.
            h: A complex indicating the phase constant.
            a: A complex indicating the coefficient of TE-component
            b: A complex indicating the coefficient of TM-component
        Returns:
            Fvec: An array of complexes [ex, ey, ez, hx, hy, hz].
        """
        pol, n, m = alpha
        u = self.samples.u(h ** 2, w, self.fill(w))
        v = self.samples.v(h ** 2, w, self.clad(w))
        gd = u / (self.r / 2)
        gm = v / (self.r / 2)
        if pol == 'E':
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
                    B_A = np.exp(v) * np.cos(u)
                    ey = a * B_A * np.exp(-gm * abs(x))
                    hx = y_te * ey
                    hz = 1j * gm / w * x / abs(x) * ey
            else:
                # parity odd
                if abs(x) <= self.r / 2:
                    ey = a * np.sin(gd * x)
                    hx = y_te * ey
                    hz = -1j * gd / w * a * np.cos(gd * x)
                else:
                    B_A = np.exp(v) * np.sin(u)
                    ey = a * B_A * x / abs(x) * np.exp(-gm * abs(x))
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
                    B_A = u / v * np.exp(v) * np.sin(u)
                    ex = b * B_A * np.exp(-gm * abs(x))
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
                    B_A = -u / v * np.exp(v) * np.cos(u)
                    ex = b * B_A * x / abs(x) * np.exp(-gm * abs(x))
                    hy = y_tm * ex
                    ez = -1j * gm * x / abs(x) / h * ex
        return np.array([ex, ey, ez, hx, hy, hz])

    def efield(self, x, y, w, l, alpha, h, a, b):
        """Return the electric field vector for the specified mode and
        point

        Args:
            x: A float indicating the x coordinate [um]
            y: A float indicating the y coordinate [um]
            w: A complex indicating the angular frequency
            l: "h" (horizontal polarization) or "v" (vertical polarization).
                In the slit case, l='h' for TM and l='v' for TE.
            alpha: A tuple (pol, n, m) where pol is 'M' for TM mode or
                'E' for TE mode, n is the order of the mode, and m is
                the number of modes in the order and the polarization,
                which is always 1 in the slit case.
            h: A complex indicating the phase constant.
            a: A complex indicating the coefficient of TE-component
            b: A complex indicating the coefficient of TM-component
        Returns:
            Evec: An array of complexes [ex, ey, ez].
        """
        pol, n, m = alpha
        u = self.samples.u(h ** 2, w, self.fill(w))
        v = self.samples.v(h ** 2, w, self.clad(w))
        gd = u / (self.r / 2)
        gm = v / (self.r / 2)
        if pol == 'E':
            ex = ez = 0.0
            if n % 2 == 1:
                # parity even
                if abs(x) <= self.r / 2:
                    ey = a * np.cos(gd * x)
                else:
                    B_A = np.exp(v) * np.cos(u)
                    ey = a * B_A * np.exp(-gm * abs(x))
            else:
                # parity odd
                if abs(x) <= self.r / 2:
                    ey = a * np.sin(gd * x)
                else:
                    B_A = np.exp(v) * np.sin(u)
                    ey = a * B_A * x / abs(x) * np.exp(-gm * abs(x))
        else:
            ey = 0.0
            if n % 2 == 0:
                # parity even
                if abs(x) <= self.r / 2:
                    ex = b * np.cos(gd * x)
                    ez = -1j * gd / h * b * np.sin(gd * x)
                else:
                    B_A = u / v * np.exp(v) * np.sin(u)
                    ex = b * B_A * np.exp(-gm * abs(x))
                    ez = (-1j * gm * x / abs(x) / h * b * B_A *
                          np.exp(-gm * abs(x)))
            else:
                # parity odd
                if abs(x) <= self.r / 2:
                    ex = b * np.sin(gd * x)
                    ez = 1j * gd / h * b * np.cos(gd * x)
                else:
                    B_A = -u / v * np.exp(v) * np.cos(u)
                    ex = b * B_A * x / abs(x) * np.exp(-gm * abs(x))
                    ez = -1j * gm / h * b * B_A * np.exp(-gm * abs(x))
        return np.array([ex, ey, ez])

    def hfield(self, x, y, w, l, alpha, h, a, b, polar=False):
        """Return the magnetic field vectors for the specified mode and
        point

        Args:
            x: A float indicating the x coordinate [um]
            y: A float indicating the y coordinate [um]
            w: A complex indicating the angular frequency
            l: "h" (horizontal polarization) or "v" (vertical polarization).
                In the slit case, l='h' for TM and l='v' for TE.
            alpha: A tuple (pol, n, m) where pol is 'M' for TM mode or
                'E' for TE mode, n is the order of the mode, and m is
                the number of modes in the order and the polarization,
                which is always 1 in the slit case.
            h: A complex indicating the phase constant.
            a: A complex indicating the coefficient of TE-component
            b: A complex indicating the coefficient of TM-component
            polar: A boolean being True if the vector should be represented by
                cylindrical polar coordinates.
        Returns:
            Hvec: An array of complexes [hx, hy, hz].
        """
        pol, n, m = alpha
        u = self.samples.u(h ** 2, w, self.fill(w))
        v = self.samples.v(h ** 2, w, self.clad(w))
        gd = u / (self.r / 2)
        gm = v / (self.r / 2)
        if pol == 'E':
            y_te = self.y_te(w, h)
            hy = 0.0
            if n % 2 == 1:
                # parity even
                if abs(x) <= self.r / 2:
                    hx = y_te * a * np.cos(gd * x)
                    hz = 1j * gd / w * a * np.sin(gd * x)
                else:
                    B_A = np.exp(v) * np.cos(u)
                    hx = y_te * a * B_A * np.exp(-gm * abs(x))
                    hz = (1j * gm / w * x / abs(x) *
                          a * B_A * np.exp(-gm * abs(x)))
            else:
                # parity odd
                if abs(x) <= self.r / 2:
                    hx = y_te * a * np.sin(gd * x)
                    hz = -1j * gd / w * a * np.cos(gd * x)
                else:
                    B_A = np.exp(v) * np.sin(u)
                    hx = y_te * a * B_A * x / abs(x) * np.exp(-gm * abs(x))
                    hz = (1j * gm / w * x / abs(x) *
                          a * B_A * x / abs(x) * np.exp(-gm * abs(x)))
        else:
            hx = hz = 0.0
            if n % 2 == 0:
                # parity even
                if abs(x) <= self.r / 2:
                    y_tm = self.y_tm_inner(w, h)
                    hy = y_tm * b * np.cos(gd * x)
                else:
                    y_tm = self.y_tm_outer(w, h)
                    B_A = u / v * np.exp(v) * np.sin(u)
                    hy = y_tm * b * B_A * np.exp(-gm * abs(x))
            else:
                # parity odd
                if abs(x) <= self.r / 2:
                    y_tm = self.y_tm_inner(w, h)
                    hy = y_tm * b * np.sin(gd * x)
                else:
                    y_tm = self.y_tm_outer(w, h)
                    B_A = -u / v * np.exp(v) * np.cos(u)
                    hy = y_tm * b * B_A * x / abs(x) * np.exp(-gm * abs(x))
        return np.array([hx, hy, hz])

    def plot_efield(self, w, l, alpha, xmax=0.5, ymax=0.25):
        """Plot the electric field distribution in the cross section.

        Args:
            w: A complex indicating the angular frequency
            l: "h" (horizontal polarization) or "v" (vertical polarization).
                In the slit case, l='h' for TM and l='v' for TE.
            alpha: A tuple (pol, n, m) where pol is 'M' for TM mode or
                'E' for TE mode, n is the order of the mode, and m is
                the number of modes in the order and the polarization,
                which is always 1 in the slit case.
            xmax: A float indicating the maximum x coordinate in the figure
            ymax: A float indicating the maximum y coordinate in the figure
        """
        import matplotlib.pyplot as plt
        xs = np.linspace(-xmax, xmax, 129)
        ys = np.linspace(-ymax, ymax, 65)
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        h = self.beta(w, alpha)
        a, b = self.coef(h, w, alpha)
        E = np.array(
            [[self.efield(x, y, w, l, alpha, h, a, b) for y in ys]
             for x in xs])
        Ex = E[:, :, 0]
        Ey = E[:, :, 1]
        Ez = E[:, :, 2]
        Es = np.sqrt(np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2)
        Emaxabs = Es.max()
        Es /= Emaxabs
        if l == 'h':
            Ex_on_x = Ex[:, 64]
            Ex_max = Ex_on_x[np.abs(Ex_on_x).argmax()]
            Enorm = Ex_max.conjugate() / abs(Ex_max) / Emaxabs
        else:
            Ey_on_x = Ey[:, 64]
            Ey_max = Ey_on_x[np.abs(Ey_on_x).argmax()]
            Enorm = Ey_max.conjugate() / abs(Ey_max) / Emaxabs
        Ex = (Ex * Enorm).real
        Ey = (Ey * Enorm).real
        Ez = (Ez * Enorm).real
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        pc = ax.pcolormesh(X, Y, Es, shading='gouraud')
        ax.quiver(X[2::5, 2::5], Y[2::5, 2::5], Ex[2::5, 2::5], Ey[2::5, 2::5],
                  scale=16.0, width=0.006, color='k',
                  pivot='middle')
        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(-ymax, ymax)
        ax.set_xlabel(r"$x\ [\mu\mathrm{m}]$", size=20)
        ax.set_ylabel(r"$y\ [\mu\mathrm{m}]$", size=20)
        plt.tick_params(labelsize=18)
        cbar = plt.colorbar(pc, shrink=0.5)
        cbar.ax.tick_params(labelsize=14)
        plt.tight_layout()
        plt.show()

    def plot_hfield(self, w, l, alpha, xmax=0.5, ymax=0.25):
        """Plot the magnetic field distribution in the cross section.

        Args:
            w: A complex indicating the angular frequency
            l: "h" (horizontal polarization) or "v" (vertical polarization).
                In the slit case, l='h' for TM and l='v' for TE.
            alpha: A tuple (pol, n, m) where pol is 'M' for TM mode or
                'E' for TE mode, n is the order of the mode, and m is
                the number of modes in the order and the polarization,
                which is always 1 in the slit case.
            xmax: A float indicating the maximum x coordinate in the figure
            ymax: A float indicating the maximum y coordinate in the figure
        """
        import matplotlib.pyplot as plt
        xs = np.linspace(-xmax, xmax, 129)
        ys = np.linspace(-ymax, ymax, 65)
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        h = self.beta(w, alpha)
        a, b = self.coef(h, w, alpha)
        H = np.array(
            [[self.hfield(x, y, w, l, alpha, h, a, b) for y in ys]
             for x in xs])
        Hx = H[:, :, 0]
        Hy = H[:, :, 1]
        Hz = H[:, :, 2]
        Hs = np.sqrt(np.abs(Hx) ** 2 + np.abs(Hy) ** 2 + np.abs(Hz) ** 2)
        Hmaxabs = Hs.max()
        Hs /= Hmaxabs
        if l == 'h':
            Hy_on_x = Hy[:, 64]
            Hy_max = Hy_on_x[np.abs(Hy_on_x).argmax()]
            Hnorm = Hy_max.conjugate() / abs(Hy_max) / Hmaxabs
        else:
            Hx_on_x = Hx[:, 64]
            Hx_max = Hx_on_x[np.abs(Hx_on_x).argmax()]
            Hnorm = Hx_max.conjugate() / abs(Hx_max) / Hmaxabs
        Hx = (Hx * Hnorm).real
        Hy = (Hy * Hnorm).real
        Hz = (Hz * Hnorm).real
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        pc = ax.pcolormesh(X, Y, Hs, shading='gouraud')
        ax.quiver(X[2::5, 2::5], Y[2::5, 2::5], Hx[2::5, 2::5], Hy[2::5, 2::5],
                  scale=16.0, width=0.006, color='k',
                  pivot='middle')
        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(-ymax, ymax)
        ax.set_xlabel(r"$x\ [\mu\mathrm{m}]$", size=20)
        ax.set_ylabel(r"$y\ [\mu\mathrm{m}]$", size=20)
        plt.tick_params(labelsize=18)
        cbar = plt.colorbar(pc, shrink=0.5)
        cbar.ax.tick_params(labelsize=14)
        plt.tight_layout()
        plt.show()

    def plot_efield_on_x_axis(self, w, l, alpha, comp, xmax=0.3, nx=128):
        """Plot a component of the electric field on the x axis

        Args:
            w: A complex indicating the angular frequency
            l: "h" (horizontal polarization) or "v" (vertical polarization).
                In the slit case, l='h' for TM and l='v' for TE.
            alpha: A tuple (pol, n, m) where pol is 'M' for TM mode or
                'E' for TE mode, n is the order of the mode, and m is
                the number of modes in the order and the polarization,
                which is always 1 in the slit case.
            comp: "x", "y" or "z" indicating the component to be drawn.
            xmax: A float indicating the maximum x coordinate in the figure
            nx: An integer indicating the number of calculational points
                (default: 128)
        """
        import matplotlib.pyplot as plt
        xs = np.linspace(-xmax, xmax, nx + 1)
        ys = np.linspace(-xmax, xmax, nx + 1)
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        h = self.beta(w, alpha)
        a, b = self.coef(h, w, alpha)
        F = np.array(
            [[self.fields(x, y, w, l, alpha, h, a, b) for y in ys]
             for x in xs])
        Ex = F[:, :, 0]
        Ey = F[:, :, 1]
        Ez = F[:, :, 2]
        Es = np.sqrt(np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2)
        Emaxabs = Es.max()
        if l == 'h':
            Ex_on_x = Ex[:, 64]
            Ex_max = Ex_on_x[np.abs(Ex_on_x).argmax()]
            Enorm = Ex_max.conjugate() / abs(Ex_max) / Emaxabs
        else:
            Ey_on_x = Ey[:, 64]
            Ey_max = Ey_on_x[np.abs(Ey_on_x).argmax()]
            Enorm = Ey_max.conjugate() / abs(Ey_max) / Emaxabs
        Ex = (Ex * Enorm)[:, nx // 2].real
        Ey = (Ey * Enorm)[:, nx // 2].real
        Ez = (Ez * Enorm)[:, nx // 2].real
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if comp == 'x':
            ax.plot(xs, Ex, "o-")
            ax.set_ylabel(r"$E_x$", size=20)
        elif comp == 'y':
            ax.plot(xs, Ey, "o-")
            ax.set_ylabel(r"$E_y$", size=20)
        elif comp == 'z':
            ax.plot(xs, Ez, "o-")
            ax.set_ylabel(r"$E_z$", size=20)
        else:
            raise ValueError('comp must be x, y or z')
        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(-1, 1)
        ax.set_xlabel(r"$x\ [\mu\mathrm{m}]$", size=20)
        plt.tick_params(labelsize=18)
        plt.tight_layout()
        plt.show()

    def plot_hfield_on_x_axis(self, w, l, alpha, comp, xmax=0.3, nx=128):
        """Plot a component of the magnetic field on the x axis

        Args:
            w: A complex indicating the angular frequency
            l: "h" (horizontal polarization) or "v" (vertical polarization).
                In the slit case, l='h' for TM and l='v' for TE.
            alpha: A tuple (pol, n, m) where pol is 'M' for TM mode or
                'E' for TE mode, n is the order of the mode, and m is
                the number of modes in the order and the polarization,
                which is always 1 in the slit case.
            comp: "x", "y" or "z" indicating the component to be drawn.
            xmax: A float indicating the maximum x coordinate in the figure
            nx: An integer indicating the number of calculational points
                (default: 128)
        """
        import matplotlib.pyplot as plt
        xs = np.linspace(-xmax, xmax, nx + 1)
        ys = np.linspace(-xmax, xmax, nx + 1)
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        h = self.beta(w, alpha)
        a, b = self.coef(h, w, alpha)
        F = np.array(
            [[self.fields(x, y, w, l, alpha, h, a, b) for y in ys]
             for x in xs])
        Hx = F[:, :, 3]
        Hy = F[:, :, 4]
        Hz = F[:, :, 5]
        Hs = np.sqrt(np.abs(Hx) ** 2 + np.abs(Hy) ** 2 + np.abs(Hz) ** 2)
        Hmaxabs = Hs.max()
        if l == 'h':
            Hy_on_x = Hy[:, 64]
            Hy_max = Hy_on_x[np.abs(Hy_on_x).argmax()]
            Hnorm = Hy_max.conjugate() / abs(Hy_max) / Hmaxabs
        else:
            Hx_on_x = Hx[:, 64]
            Hx_max = Hx_on_x[np.abs(Hx_on_x).argmax()]
            Hnorm = Hx_max.conjugate() / abs(Hx_max) / Hmaxabs
        Hx = (Hx * Hnorm)[:, nx // 2].real
        Hy = (Hy * Hnorm)[:, nx // 2].real
        Hz = (Hz * Hnorm)[:, nx // 2].real
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if comp == 'x':
            ax.plot(xs, Hx, "o-")
            ax.set_ylabel(r"$H_x$", size=20)
        elif comp == 'y':
            ax.plot(xs, Hy, "o-")
            ax.set_ylabel(r"$H_y$", size=20)
        elif comp == 'z':
            ax.plot(xs, Hz, "o-")
            ax.set_ylabel(r"$H_z$", size=20)
        else:
            raise ValueError('comp must be x, y or z')
        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(-1, 1)
        ax.set_xlabel(r"$x\ [\mu\mathrm{m}]$", size=20)
        plt.tick_params(labelsize=18)
        plt.tight_layout()
        plt.show()

    def plot_betas(self, fmin=2.5, fmax=5.5, wi=0.0, comp='imag', nw=128):
        """Plot propagation constants as a function of frequency

        Args:
            fmin: A float indicating the minimum frequency [100THz]
                (default: 2.5)
            fmax: A float indicating the maximum frequency [100THz]
                (default: 2.5)
            wi: A float indicating the imaginary part of angular frequency
            comp: "real" (phase constants) or "imag" (attenuation constants)
            nw: An integer indicating the number of calculational points
                within the frequency range (default: 128).
        """
        from scipy.constants import c
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fs = np.linspace(fmin, fmax, nw + 1)
        ws = fs * 2 * np.pi / (c * 1e-8)
        markers = ["o", "s", "^", "v", "d", "p", "*"]
        for alpha in self.alpha_list:
            pol, n, m = alpha
            label = self.labels[alpha]
            if pol == 'E':
                if m == 1:
                    mfc = "k"
                else:
                    mfc = "0.3"
            else:
                if m == 1:
                    mfc = "w"
                else:
                    mfc = "0.7"
            marker = markers[n]
            if comp == 'real':
                hs = [self.beta(wr + 1j * wi, alpha).real for wr in ws]
            elif comp == 'imag':
                hs = [self.beta(wr + 1j * wi, alpha).imag for wr in ws]
            elif comp == 'gamma2':
                hs = []
                for wr in ws:
                    w = wr + 1j * wi
                    hs.append(
                        (self.beta(w, alpha).real -
                         self.clad(w) * w ** 2).real)
            else:
                raise ValueError("comp must be 'real', 'imag' or 'gamma2'.")
            ax.plot(fs, hs, "k-", linewidth=2)
            ax.plot(fs[::4], hs[::4], label=label, marker=marker,
                    markerfacecolor=mfc,
                    linestyle="None", color="k", markersize=8,
                    markeredgewidth=2, linewidth=2)
        for alpha in self.alpha_list:
            pol, n, m = alpha
            label = self.labels[alpha]
            if pol == 'E':
                if m == 1:
                    mfc = "k"
                else:
                    mfc = "0.3"
            else:
                if m == 1:
                    mfc = "w"
                else:
                    mfc = "0.7"
            marker = markers[n]
            if comp == 'real':
                hs_pec = [self.beta_pec(wr + 1j * wi, alpha).real for wr in ws]
            elif comp == 'imag':
                hs_pec = [self.beta_pec(wr + 1j * wi, alpha).imag for wr in ws]
            elif comp == 'gamma2':
                hs_pec = []
                for wr in ws:
                    w = wr + 1j * wi
                    hs_pec.append(
                        (self.beta_pec(w, alpha).real -
                         self.clad(w) * w ** 2).real)
            else:
                raise ValueError("comp must be 'real', 'imag' or 'gamma2'.")
            if pol == 'M':
                ax.plot(fs, hs_pec, "b-", linewidth=2)
                ax.plot(fs[::4], hs_pec[::4], label="PEC{0}".format(n),
                        marker=marker,
                        markerfacecolor='b',
                        linestyle="None", color="b", markersize=8,
                        markeredgewidth=2, linewidth=2)
        ax.set_xlim(fs[0], fs[-1])
        ax.set_xlabel(r'$\nu$ $[\mathrm{100THz}]$', size=20)
        if comp == 'imag':
            ax.set_ylabel(r'attenuation constant', size=20)
        else:
            ax.set_ylabel(r'phase constant', size=20)
        plt.tick_params(labelsize=18)
        plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
        fig.subplots_adjust(left=0.11, right=0.83, bottom=0.12, top=0.97)
        plt.show()

    def coefs_numpy(self, hs, w):
        As = []
        Bs = []
        for h, s, n, m in zip(hs, self.s_all, self.n_all, self.m_all):
            pol = 'E' if s == 0 else 'M'
            ai, bi = self.coef(h, w, (pol, n, m))
            As.append(ai)
            Bs.append(bi)
        return np.ascontiguousarray(As), np.ascontiguousarray(Bs)

    def coefs(self, hs, w):
        return coefs_cython(self, hs, w)

    def Ys(self, w, hs, As, Bs):
        vals = []
        for h, s, n, a, b in zip(hs, self.s_all, self.n_all, As, Bs):
            pol = 'E' if s == 0 else 'M'
            vals.append(self.Y(w, h, (pol, n, 1), a, b))
        return np.array(vals)

    def hAB(self, w):
        hs = np.array([self.beta(w, alpha)
                       for alpha in self.alpha_all])
        As, Bs = self.coefs(hs, w)
        return hs, As, Bs

    def ABY(self, w, hs):
        e1 = self.fill(w)
        e2 = self.clad(w)
        return ABY_cython(
            w, self.r, self.s_all, self.n_all, hs, e1, e2)

    def hABY(self, w):
        e1 = self.fill(w)
        e2 = self.clad(w)
        hs = np.array([self.beta(w, alpha)
                       for alpha in self.alpha_all])
        As, Bs, Y = ABY_cython(
            w, self.r, self.s_all, self.n_all, hs, e1, e2)
        return hs, As, Bs, Y
