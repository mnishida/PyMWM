# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import jv, jvp, kv, kvp, jn_zeros, jnp_zeros


class Cylinder(object):
    """A class defining a cylindrical waveguide.

     Attributes:
        fill: An instance of Material class for the core
        clad: An instance of Material class for the clad
        r: A float indicating the radius of the circular cross section [um].
        samples: An instance of Samples class
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
                    'num_m': An integer indicating the number of modes in each
                        order and polarization.
        """
        from pyoptmat import Material
        from pymwm.cylinder.samples import Samples
        self.r = params['core']['size']
        self.fill = Material(params['core']['fill'])
        self.clad = Material(params['clad'])
        self.samples = Samples(self.r, self.fill, self.clad, params['modes'])
        self.bounds = params['bounds']
        try:
            betas, convs = self.samples.load()
        except:
            from multiprocessing import Pool
            num_n = params['modes']['num_n']
            p = Pool(num_n)
            betas_list = p.map(self.samples, range(num_n))
            betas = {key: val for betas, convs in betas_list
                     for key, val in betas.items()}
            convs = {key: val for betas, convs in betas_list
                     for key, val in convs.items()}
            self.samples.save(betas, convs)
        self.beta_funcs = self.samples.interpolation(betas, convs, self.bounds)
        self.alpha_list = []
        for alpha, comp in self.beta_funcs.keys():
            if comp == 'real':
                self.alpha_list.append(alpha)
        self.alpha_list.sort()
        self.labels = {}
        for alpha in self.alpha_list:
            pol, n, m = alpha
            if pol == 'E':
                if n == 0:
                    self.labels[alpha] = (
                        r'TE$_{0' + r'{0}'.format(m) + r'}$')
                else:
                    self.labels[alpha] = (
                        r'HE$_{' + r'{0}{1}'.format(n, m) + r'}$')
            else:
                if n == 0:
                    self.labels[alpha] = (
                        r'TM$_{0' + r'{0}'.format(m) + r'}$')
                else:
                    self.labels[alpha] = (
                        r'EH$_{' + r'{0}{1}'.format(n, m) + r'}$')

    def beta(self, w, alpha):
        """Return phase constant

        Args:
            w: A complex indicating the angular frequency
            alpha: A tuple (pol, n, m) where pol is 'M' for TM-like mode or
                'E' for TE-like mode, n is the order of the mode, and m is
                the number of modes in the order and the polarization.
        Returns:
            h: A complex indicating the phase constant.
        """
        wr = w.real
        wi = w.imag
        hr = self.beta_funcs[(alpha, 'real')](wr, wi)[0, 0]
        hi = self.beta_funcs[(alpha, 'imag')](wr, wi)[0, 0]
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
        wcomp = w.real + 1j * w.imag
        pol, n, m = alpha
        if pol == "E":
            chi = jnp_zeros(n, m)[-1]
        elif pol == "M":
            chi = jn_zeros(n, m)[-1]
        val = np.sqrt(self.fill(wcomp) * wcomp ** 2 - chi ** 2 / self.r ** 2)
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
        if self.clad.model == 'pec':
            if pol == 'E':
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
            ci = - n * (u ** 2 + v ** 2) * jnu * knv / (u * v)
            if pol == 'E':
                ci *= (h / w) ** 2
                ci /= (e1 * jnpu * v * knv + e2 * knpv * u * jnu)
                norm = self.norm(w, h, alpha, 1.0 + 0.0j, ci)
                ai = 1.0 / norm
                bi = ci / norm
            else:
                ci /= (jnpu * v * knv + knpv * u * jnu)
                norm = self.norm(w, h, alpha, ci, 1.0 + 0.0j)
                bi = 1.0 / norm
                ai = ci / norm
        return ai, bi

    def norm(self, w, h, alpha, a, b):
        pol, n, m = alpha
        en = 1 if n == 0 else 2
        if self.clad(w).real < -1e6:
            radius = self.r
            if pol == 'E':
                u = jnp_zeros(n, m)[-1]
                jnu = jv(n, u)
                jnpu = 0.0
            else:
                u = jn_zeros(n, m)[-1]
                jnu = 0.0
                jnpu = jvp(n, u)
            return np.sqrt(np.abs(a) ** 2 * np.pi * radius ** 2 / en *
                           (1 - n ** 2 / u ** 2) * jnu ** 2 +
                           np.abs(b) ** 2 * np.pi * radius ** 2 / en *
                           jnpu ** 2)
        ac = a.conjugate()
        bc = b.conjugate()
        u = self.samples.u(h ** 2, w, self.fill(w))
        v = self.samples.v(h ** 2, w, self.clad(w))
        jnu = jv(n, u)
        jnpu = jvp(n, u)
        knv = kv(n, v)
        knpv = kvp(n, v)
        val_u = 2 * np.pi * self.r ** 2 / en
        val_v = val_u * np.abs((u * jnu) / (v * knv)) ** 2
        upart_diag = self.upart_diag(n, u, jnu, jnpu)
        vpart_diag = self.vpart_diag(n, v, knv, knpv)
        upart_off = self.upart_off(n, u, jnu)
        vpart_off = self.vpart_off(n, v, knv)
        return np.sqrt(np.real(
            val_u * (
                a * (ac * upart_diag + bc * upart_off) +
                b * (bc * upart_diag + ac * upart_off)) -
            val_v * (
                a * (ac * vpart_diag + bc * vpart_off) +
                b * (bc * vpart_diag + ac * vpart_off))))

    def upart_diag(self, n, u, jnu, jnpu):
        if abs(u.imag) < 1e-16:
            return (jnu * jnpu / u + (
                jnpu ** 2 + (1 - n ** 2 / u ** 2) * jnu ** 2) / 2)
        if abs(u.real) < 1e-16:
            return (-1) ** (n - 1) * (
                jnu * jnpu / u + (
                    jnpu ** 2 + (1 - n ** 2 / u ** 2) * jnu ** 2) / 2)
        uc = u.conjugate()
        jnuc = jnu.conjugate()
        jnpuc = jnpu.conjugate()
        return (uc * jnuc * jnpu -
                u * jnu * jnpuc) / (uc ** 2 - u ** 2)

    def upart_off(self, n, u, jnu):
        return n * np.abs(jnu / u) ** 2

    def vpart_diag(self, n, v, knv, knpv):
        if abs(v.imag) < 1e-16:
            return (knv * knpv / v + (
                knpv ** 2 - (1 + n ** 2 / v ** 2) * knv ** 2) / 2)
        if abs(v.real) < 1e-16:
            return (-1) ** (n - 1) * (
                knv * knpv / v + (
                    knpv ** 2 - (1 + n ** 2 / v ** 2) * knv ** 2) / 2)
        vc = v.conjugate()
        knvc = knv.conjugate()
        knpvc = knpv.conjugate()
        return (vc * knvc * knpv -
                v * knv * knpvc) / (vc ** 2 - v ** 2)

    def vpart_off(self, n, v, knv):
        return n * np.abs(knv / v) ** 2

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
            if pol == 'E':
                val = h / w
            else:
                val = e1 * w / h
        else:
            ac = a.conjugate()
            bc = b.conjugate()
            u = self.samples.u(h ** 2, w, e1)
            v = self.samples.v(h ** 2, w, e2)
            jnu = jv(n, u)
            jnpu = jvp(n, u)
            knv = kv(n, v)
            knpv = kvp(n, v)
            val_u = 2 * np.pi * self.r ** 2 / en
            val_v = val_u * np.abs((u * jnu) / (v * knv)) ** 2
            upart_diag = self.upart_diag(n, u, jnu, jnpu)
            vpart_diag = self.vpart_diag(n, v, knv, knpv)
            upart_off = self.upart_off(n, u, jnu)
            vpart_off = self.vpart_off(n, v, knv)
            val = (val_u * (h / w * a *
                            (ac * upart_diag + bc * upart_off) +
                            e1 * w / h * b *
                            (bc * upart_diag + ac * upart_off)) -
                   val_v * (h / w * a *
                            (ac * vpart_diag + bc * vpart_off) +
                            e2 * w / h * b *
                            (bc * vpart_diag + ac * vpart_off)))
        return val

    def y_te(self, w, h):
        return h / w

    def y_tm_inner(self, w, h):
        e = self.fill(w)
        return e * w / h

    def y_tm_outer(self, w, h):
        e = self.clad(w)
        return e * w / h

    def efield(self, x, y, w, l, alpha, h, a, b, polar=False):
        """Return the electric field vector for the specified mode and point

        Args:
            x: A float indicating the x coordinate [um]
            y: A float indicating the y coordinate [um]
            w: A complex indicating the angular frequency
            l: "h" (horizontal polarization) or "v" (vertical polarization)
            alpha: A tuple (pol, n, m) where pol is 'M' for TM-like mode or
                'E' for TE-like mode, n is the order of the mode, and m is
                the number of modes in the order and the polarization.
            h: A complex indicating the phase constant.
            a: A complex indicating the coefficient of TE-component
            b: A complex indicating the coefficient of TM-component
            polar: A boolean being True if the vector should be represented by
                cylindrical polar coordinates.
        Returns:
            Evec: An array of complexes [ex, ey, ez].
        """
        pol, n, m = alpha
        r = np.hypot(x, y)
        p = np.arctan2(y, x)
        u = self.samples.u(h ** 2, w, self.fill(w))
        v = self.samples.v(h ** 2, w, self.clad(w))
        ur = u * r / self.r
        vr = v * r / self.r
        if l == 'h':
            fr = np.cos(n * p)
            fp = -np.sin(n * p)
        else:
            fr = np.sin(n * p)
            fp = np.cos(n * p)
        if r <= self.r:
            er = (a * (jv(n - 1, ur) + jv(n + 1, ur)) / 2 +
                  b * jvp(n, ur)) * fr
            ep = (a * jvp(n, ur) +
                  b * (jv(n - 1, ur) + jv(n + 1, ur)) / 2) * fp
            ez = u / (self.r * 1j * h) * b * jv(n, ur) * fr
        else:
            val = - u * jv(n, u) / (v * kv(n, v))
            er = (- a * (kv(n - 1, vr) - kv(n + 1, vr)) / 2 +
                  b * kvp(n, vr)) * fr * val
            ep = (a * kvp(n, vr) -
                  b * (kv(n - 1, vr) - kv(n + 1, vr)) / 2) * fp * val
            ez = - v / (self.r * 1j * h) * b * kv(n, vr) * fr * val
        if polar:
            return np.array([er, ep, ez])
        else:
            ex = er * np.cos(p) - ep * np.sin(p)
            ey = er * np.sin(p) + ep * np.cos(p)
            return np.array([ex, ey, ez])

    def hfield(self, x, y, w, l, alpha, h, a, b, polar=False):
        """Return the magnetic field vector for the specified mode and point

        Args:
            x: A float indicating the x coordinate [um]
            y: A float indicating the y coordinate [um]
            w: A complex indicating the angular frequency
            l: "h" (horizontal polarization) or "v" (vertical polarization)
            alpha: A tuple (pol, n, m) where pol is 'M' for TM-like mode or
                'E' for TE-like mode, n is the order of the mode, and m is
                the number of modes in the order and the polarization.
            h: A complex indicating the phase constant.
            a: A complex indicating the coefficient of TE-component
            b: A complex indicating the coefficient of TM-component
            polar: A boolean being True if the vector should be represented by
                cylindrical polar coordinates.
        Returns:
            Hvec: An array of complexes [hx, hy, hz].
        """
        pol, n, m = alpha
        r = np.hypot(x, y)
        p = np.arctan2(y, x)
        u = self.samples.u(h ** 2, w, self.fill(w))
        v = self.samples.v(h ** 2, w, self.clad(w))
        ur = u * r / self.r
        vr = v * r / self.r
        if l == 'h':
            fr = np.cos(n * p)
            fp = -np.sin(n * p)
        else:
            fr = np.sin(n * p)
            fp = np.cos(n * p)
        if r <= self.r:
            y_te = self.y_te(w, h)
            y_tm = self.y_tm_inner(w, h)
            hr = -(a * y_te * jvp(n, ur) +
                   b * y_tm * (jv(n - 1, ur) + jv(n + 1, ur)) / 2) * fp
            hp = (a * y_te * (jv(n - 1, ur) + jv(n + 1, ur)) / 2 +
                  b * y_tm * jvp(n, ur)) * fr
            hz = - u / self.r * a * jv(n, ur) * fr
        else:
            y_te = self.y_te(w, h)
            y_tm = self.y_tm_outer(w, h)
            val = - u * jv(n, u) / (v * kv(n, v))
            hr = -(a * y_te * kvp(n, vr) -
                   b * y_tm * (kv(n - 1, vr) - kv(n + 1, vr)) / 2) * fp * val
            hp = (- a * y_te * (kv(n - 1, vr) - kv(n + 1, vr)) / 2 +
                  b * y_tm * kvp(n, vr)) * fr * val
            hz = v / self.r * a * kv(n, vr) * fr * val
        if polar:
            return np.array([hr, hp, hz])
        else:
            hx = hr * np.cos(p) - hp * np.sin(p)
            hy = hr * np.sin(p) + hp * np.cos(p)
            return np.array([hx, hy, hz])

    def plot_efield(self, w, l, alpha, xmax=0.25, ymax=0.25):
        """Plot the electric field distribution in the cross section.

        Args:
            w: A complex indicating the angular frequency
            l: "h" (horizontal polarization) or "v" (vertical polarization)
            alpha: A tuple (pol, n, m) where pol is 'M' for TM-like mode or
                'E' for TE-like mode, n is the order of the mode, and m is
                the number of modes in the order and the polarization.
            xmax: A float indicating the maximum x coordinate in the figure
            ymax: A float indicating the maximum y coordinate in the figure
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        xs = np.linspace(-xmax, xmax, 128)
        ys = np.linspace(-ymax, ymax, 128)
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        h = self.beta(w, alpha)
        a, b = self.coef(h, w, alpha)
        E = np.array(
            [[self.efield(x, y, w, l, alpha, h, a, b, False) for y in ys]
             for x in xs])
        Ex = E[:, :, 0].real
        Ey = E[:, :, 1].real
        Ez = E[:, :, 2].real
        Es = np.sqrt(np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2)
        Emax = Es.max()
        Ex /= Emax
        Ey /= Emax
        Ez /= Emax
        Es /= Emax
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        pc = ax.pcolormesh(X, Y, Es, shading='gouraud')
        circle = Circle((0.0, 0.0), self.r, fill=False, ls='solid',
                        color='w')
        ax.add_patch(circle)
        ax.quiver(X[2::5, 2::5], Y[2::5, 2::5], Ex[2::5, 2::5], Ey[2::5, 2::5],
                  scale=16.0, width=0.006, color='k',
                  pivot='middle')
        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(-ymax, ymax)
        ax.set_xlabel(r"$x\ [\mu\mathrm{m}]$", size=20)
        ax.set_ylabel(r"$y\ [\mu\mathrm{m}]$", size=20)
        plt.tick_params(labelsize=18)
        cbar = plt.colorbar(pc)
        cbar.ax.tick_params(labelsize=14)
        plt.tight_layout()
        plt.show()

    def plot_hfield(self, w, l, alpha, xmax=0.25, ymax=0.25):
        """Plot the magnetic field distribution in the cross section.

        Args:
            w: A complex indicating the angular frequency
            l: "h" (horizontal polarization) or "v" (vertical polarization)
            alpha: A tuple (pol, n, m) where pol is 'M' for TM-like mode or
                'E' for TE-like mode, n is the order of the mode, and m is
                the number of modes in the order and the polarization.
            xmax: A float indicating the maximum x coordinate in the figure
            ymax: A float indicating the maximum y coordinate in the figure
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        xs = np.linspace(-xmax, xmax, 128)
        ys = np.linspace(-ymax, ymax, 128)
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        h = self.beta(w, alpha)
        a, b = self.coef(h, w, alpha)
        H = np.array(
            [[self.hfield(x, y, w, l, alpha, h, a, b, False) for y in ys]
             for x in xs])
        Hx = H[:, :, 0].real
        Hy = H[:, :, 1].real
        Hz = H[:, :, 2].real
        Hs = np.sqrt(np.abs(Hx) ** 2 + np.abs(Hy) ** 2 + np.abs(Hz) ** 2)
        Hmax = Hs.max()
        Hx /= Hmax
        Hy /= Hmax
        Hz /= Hmax
        Hs /= Hmax
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        pc = ax.pcolormesh(X, Y, Hs, shading='gouraud')
        circle = Circle((0.0, 0.0), self.r, fill=False, ls='solid',
                        color='w')
        ax.add_patch(circle)
        ax.quiver(X[2::5, 2::5], Y[2::5, 2::5], Hx[2::5, 2::5], Hy[2::5, 2::5],
                  scale=16.0, width=0.006, color='k',
                  pivot='middle')
        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(-ymax, ymax)
        ax.set_xlabel(r"$x\ [\mu\mathrm{m}]$", size=20)
        ax.set_ylabel(r"$y\ [\mu\mathrm{m}]$", size=20)
        plt.tick_params(labelsize=18)
        cbar = plt.colorbar(pc)
        cbar.ax.tick_params(labelsize=14)
        plt.tight_layout()
        plt.show()

    def plot_efield_on_x_axis(self, w, l, alpha, comp, xmax=0.3, nx=128):
        """Plot a component of the electric field on the x axis

        Args:
            w: A complex indicating the angular frequency
            l: "h" (horizontal polarization) or "v" (vertical polarization)
            alpha: A tuple (pol, n, m) where pol is 'M' for TM-like mode or
                'E' for TE-like mode, n is the order of the mode, and m is
                the number of modes in the order and the polarization.
            comp: "x", "y" or "z" indicating the component to be drawn.
            xmax: A float indicating the maximum x coordinate in the figure
            nx: An integer indicating the number of calculational points
                (default: 128)
        """
        import matplotlib.pyplot as plt
        xs = np.linspace(-xmax, xmax, nx)
        h = self.beta(w, alpha)
        a, b = self.coef(h, w, alpha)
        E = np.array(
            [self.efield(x, 0.0, w, l, alpha, h, a, b, False)
             for x in xs])
        Ex = E[:, 0].real
        Ey = E[:, 1].real
        Ez = E[:, 2].real
        Es = np.sqrt(np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2)
        Emax = Es.max()
        Ex /= Emax
        Ey /= Emax
        Ez /= Emax
        Es /= Emax
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
        ax.set_xlabel(r"$x\ [\mu\mathrm{m}]$", size=20)
        plt.tick_params(labelsize=18)
        plt.tight_layout()
        plt.show()

    def plot_hfield_on_x_axis(self, w, l, alpha, comp, xmax=0.3, nx=128):
        """Plot a component of the magnetic field on the x axis

        Args:
            w: A complex indicating the angular frequency
            l: "h" (horizontal polarization) or "v" (vertical polarization)
            alpha: A tuple (pol, n, m) where pol is 'M' for TM-like mode or
                'E' for TE-like mode, n is the order of the mode, and m is
                the number of modes in the order and the polarization.
            comp: "x", "y" or "z" indicating the component to be drawn.
            xmax: A float indicating the maximum x coordinate in the figure
            nx: An integer indicating the number of calculational points
                (default: 128)
        """
        import matplotlib.pyplot as plt
        xs = np.linspace(-xmax, xmax, nx)
        h = self.beta(w, alpha)
        a, b = self.coef(h, w, alpha)
        H = np.array(
            [self.hfield(x, 0.0, w, l, alpha, h, a, b, False)
             for x in xs])
        Hx = H[:, 0].real
        Hy = H[:, 1].real
        Hz = H[:, 2].real
        Hs = np.sqrt(np.abs(Hx) ** 2 + np.abs(Hy) ** 2 + np.abs(Hz) ** 2)
        Hmax = Hs.max()
        Hx /= Hmax
        Hy /= Hmax
        Hz /= Hmax
        Hs /= Hmax
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
