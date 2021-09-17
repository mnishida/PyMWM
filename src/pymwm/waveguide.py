from __future__ import annotations

import abc
import os
from collections import OrderedDict
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import riip
from pandas import DataFrame


class Sampling(metaclass=abc.ABCMeta):
    """A class provides sampling methods.

    Attributes:
        fill: An instance of Material class for the core
        clad: An instance of Material class for the clad
        size: A float indicating the size of core [um].
        size2: A float indicating the optional size of core [um].
        params: A dict whose keys and values are as follows:
            'wl_max': A float indicating the maximum wavelength [um]
            'wl_min': A float indicating the minimum wavelength [um]
            'wl_imag': A float indicating the minimum value of
                abs(c / f_imag) [um] where f_imag is the imaginary part of
                the frequency.
            'dw': A float indicating frequency interval
                [rad * c / 1um]=[2.99792458e14 rad / s].
            'num_n': An integer indicating the number of orders of modes.
            'num_m': An integer indicating the number of modes in each
                order and polarization.
        ws: A 1D array indicating the real part of the angular frequencies
            to be calculated [rad (c / 1um)]=[2.99792458e14 rad / s].
        wis: A 1D array indicating the imaginary part of the angular
            frequencies to be calculated [rad * (c / 1um)].
    """

    @abc.abstractmethod
    def __init__(
        self,
        size: float,
        fill: dict,
        clad: dict,
        params: dict,
        size2: float = 0.0,
    ):
        self.size = size
        self.size2 = size2
        rid = riip.RiiDataFrame()
        self.fill = rid.material(fill)
        self.clad = rid.material(clad)
        self.params = params
        self.database = Database(self.key)
        self.ws = self.database.ws
        self.wis = self.database.wis

    @abc.abstractproperty
    def shape(self):
        pass

    @abc.abstractproperty
    def num_all(self):
        pass

    @property
    def key(self) -> dict:
        p = self.params
        dw = p.get("dw", 1.0 / 64)
        wl_max = p.get("wl_max", 5.0)
        wl_min = p.get("wl_min", 0.4)
        wl_imag = p.get("wl_imag", 5.0)
        shape = self.shape
        size = self.size
        size2 = self.size2
        num_n = p["num_n"]
        num_m = p["num_m"]
        num_all = self.num_all
        d = dict(
            (
                ("shape", shape),
                ("size", size),
                ("size2", size2),
                ("core", self.fill.label),
                ("clad", self.clad.label),
                ("wl_max", wl_max),
                ("wl_min", wl_min),
                ("wl_imag", wl_imag),
                ("dw", dw),
                ("num_n", num_n),
                ("num_m", num_m),
                ("num_all", num_all),
            )
        )
        return d

    def plot_convs(self, convs: np.ndarray, alpha: tuple[str, int, int]) -> None:
        """Show area of convergence where a solution is found in complex angular frequency plane

        Args:
            convs (np.ndarray[bool]): True if a solution is found.
            alpha (tuple[pol: str, n: int, m: int]): Label of mode specifying
                TE-like (pol="E") or TM-like (pol="M") polalization and the order (n, m).
        """
        x, y = np.meshgrid(self.ws, self.wis, indexing="ij")
        z = convs[alpha]
        plt.pcolormesh(x, y, z, shading="auto", rasterized=True)
        plt.colorbar(pad=0.02)
        plt.show()

    def plot_real_betas(self, betas: np.ndarray, alpha: tuple[str, int, int]) -> None:
        """Show a colormap of phase constant in complex angular frequency plane

        Args:
            betas (np.ndarray[bool]): Complex propagation constant.
            alpha (tuple[pol: str, n: int, m: int]): Label of mode specifying
                TE-like (pol="E") or TM-like (pol="M") polalization and the order (n, m).
        """
        x, y = np.meshgrid(self.ws, self.wis, indexing="ij")
        z = betas[alpha]
        plt.pcolormesh(x, y, z.real, shading="auto", rasterized=True)
        plt.colorbar(pad=0.02)
        plt.show()

    def plot_imag_betas(self, betas: np.ndarray, alpha: tuple[str, int, int]) -> None:
        """Show colormap of attenuation constant in complex angular frequency plane

        Args:
            betas (np.ndarray[bool]): Complex propagation constant.
            alpha (tuple[pol: str, n: int, m: int]): Label of mode specifying
                TE-like (pol="E") or TM-like (pol="M") polalization and the order (n, m).
        """
        x, y = np.meshgrid(self.ws, self.wis, indexing="ij")
        z = betas[alpha]
        plt.pcolormesh(x, y, z.imag, shading="auto", rasterized=True)
        plt.colorbar(pad=0.02)
        plt.show()


class Waveguide(metaclass=abc.ABCMeta):
    """A class defining a abstract waveguide.

    Attributes:
       fill_params: A dict of the parameters of the core Material.
       fill: An instance of Material class for the core
       clad_params: A dict of the parameters of the clad Material.
       clad: An instance of Material class for the clad
       r: A float indicating the radius of the circular cross section [um].
    """

    @abc.abstractmethod
    def __init__(self, params: dict):
        """Init abstract Waveguide class.

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
                    'wl_imag': A float indicating the minimum value of
                        abs(c / f_imag) [um] where f_imag is the imaginary part
                        of the frequency.
                    'beta_imag_max': A float indicating the maximum allowed value
                        of the imaginary part of beta.
                'modes': A dict of the settings for calculating modes:
                    'wl_max': A float indicating the maximum wavelength [um]
                        (default: 5.0)
                    'wl_min': A float indicating the minimum wavelength [um]
                        (default: 0.4)
                    'wl_imag': A float indicating the minimum value of
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
        self.r = params["core"]["size"]
        self.fill_params = params["core"]["fill"].copy()
        self.fill_params["bound_check"] = False
        self.clad_params = params["clad"].copy()
        self.clad_params["bound_check"] = False
        rid = riip.RiiDataFrame()
        self.fill = rid.material(self.fill_params)
        self.clad = rid.material(self.clad_params)
        self.num_n = params["modes"]["num_n"]
        self.num_m = params["modes"]["num_m"]
        self.bounds = params["bounds"]
        self.ls = params["modes"].get("ls", ["h", "v"])
        betas, convs, self.samples = self.betas_convs_samples(params)
        self.beta_funcs = self.samples.database.interpolation(betas, convs, self.bounds)

        alpha_list = []
        alpha_candidates = params["modes"].get("alphas", None)
        for alpha, comp in self.beta_funcs.keys():
            if comp == "real":
                if alpha_candidates is not None:
                    if alpha in alpha_candidates:
                        alpha_list.append(alpha)
                else:
                    alpha_list.append(alpha)
        alpha_list.sort()
        self.alphas = self.modes = self.get_alphas(alpha_list)

        self.alpha_all = [alpha for dir in self.ls for alpha in self.alphas[dir]]
        self.l_all = np.array(
            [0 if dir == "h" else 1 for dir in self.ls for _ in self.alphas[dir]]
        )
        self.s_all = np.array(
            [
                0 if pol == "E" else 1
                for dir in self.ls
                for pol, n, m in self.alphas[dir]
            ],
            dtype=int,
        )
        self.n_all = np.array(
            [n for dir in self.ls for pol, n, m in self.alphas[dir]], dtype=int
        )
        self.m_all = np.array(
            [m for dir in self.ls for pol, n, m in self.alphas[dir]], dtype=int
        )
        self.num_n_all = self.n_all.shape[0]

    @abc.abstractmethod
    def get_alphas(self, alpha_list: list[tuple[str, int, int]]) -> dict:
        pass

    @abc.abstractmethod
    def betas_convs_samples(
        self, params: dict
    ) -> tuple[np.ndarray, np.ndarray, Sampling]:
        pass

    @abc.abstractmethod
    def beta(self, w: complex, alpha: tuple[str, int, int]) -> complex:
        pass

    @abc.abstractmethod
    def beta_pec(self, w: complex, alpha: tuple[str, int, int]) -> complex:
        pass

    @abc.abstractmethod
    def coef(self, h: complex, w: complex, alpha: tuple[str, int, int]) -> tuple:
        pass

    @abc.abstractmethod
    def fields(
        self,
        x: float,
        y: float,
        w: complex,
        dir: str,
        alpha: tuple[str, int, int],
        h: complex,
        coef: tuple,
    ) -> np.ndarray:
        """Return the field vectors for the specified mode and point

        Args:
            x: The x coordinate [um].
            y: The y coordinate [um].
            w: The angular frequency.
            dir: "h" (horizontal polarization) or "v" (vertical polarization)
            alpha: (pol, n, m)
                pol: 'M' (TM-like mode) or 'E' (TE-like mode).
                n: The order of the mode.
                m: The sub order of the mode.
            h: The complex phase constant.
            coef: The coefficients of TE- and TM- components
        Returns:
            f_vec: An array of complexes [ex, ey, ez, hx, hy, hz].
        """
        pass

    @abc.abstractmethod
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
        """Return the field vectors for the specified mode and point

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
        pass

    @abc.abstractmethod
    def h_field(
        self,
        x: float,
        y: float,
        w: complex,
        dir: str,
        alpha: tuple[str, int, int],
        h: complex,
        coef: tuple,
    ) -> np.ndarray:
        """Return the field vectors for the specified mode and point

        Args:
            x: The x coordinate [um].
            y: The y coordinate [um].
            w: The angular frequency.
            dir: "h" (horizontal polarization) or "v" (vertical polarization)
            alpha: (pol, n, m)
                pol: 'M' (TM-like mode) or 'E' (TE-like mode).
                n: The order of the mode.
                m: The sub order of the mode.
            h: The complex phase constant.
            coef: The coefficients of TE- and TM- components
        Returns:
            h_vec: An array of complexes [hx, hy, hz].
        """
        pass

    def plot_beta(
        self,
        alpha: tuple[str, int, int],
        fmt: Optional[str] = "-",
        wl_max: float = 1.0,
        wl_min: float = 0.4,
        wi: float = 0.0,
        comp: str = "imag",
        nw: int = 128,
        **kwargs,
    ):
        """Plot propagation constants as a function of wavelength.

        Args:
            alpha: (pol, n, m)
                pol: 'E' (TE-like mode) or 'M' (TM-like mode).
                n: The order of the mode.
                m: The sub order of the mode.
            fmt: The plot format string.
            wl_max: The maximum wavelength [um].
            wl_min: The minimum wavelength [um].
            wi: The imaginary part of angular frequency.
            comp: "real" (phase constants) or "imag" (attenuation constants).
            nw: The number of calculational points within the frequency range.
        """
        wls = np.linspace(wl_max, wl_min, nw + 1)
        ws = 2 * np.pi / wls
        pol, n, m = alpha
        label = f"({pol},{n},{m})"
        if comp == "real":
            hs = [self.beta(wr + 1j * wi, alpha).real for wr in ws]
        elif comp == "imag":
            hs = [self.beta(wr + 1j * wi, alpha).imag for wr in ws]
        elif comp == "gamma2":
            hs = []
            for wr in ws:
                w = wr + 1j * wi
                hs.append((self.beta(w, alpha).real - self.clad(w) * w ** 2).real)
        else:
            raise ValueError("comp must be 'real', 'imag' or 'gamma2'.")
        (line,) = plt.plot(wls, hs, fmt, label=label, **kwargs)
        kwargs.setdefault("color", line.get_color())
        if comp == "real":
            hs_pec = [self.beta_pec(wr + 1j * wi, alpha).real for wr in ws]
        elif comp == "imag":
            hs_pec = [self.beta_pec(wr + 1j * wi, alpha).imag for wr in ws]
        elif comp == "gamma2":
            hs_pec = []
            for wr in ws:
                w = wr + 1j * wi
                hs_pec.append(
                    (self.beta_pec(w, alpha).real - self.clad(w) * w ** 2).real
                )
        else:
            raise ValueError("comp must be 'real', 'imag' or 'gamma2'.")
        plt.plot(wls, hs_pec, "--", label="PEC", **kwargs)
        plt.xlabel(r"$\lambda$ $[\mathrm{\mu m}]$")
        plt.xlim(wl_min, wl_max)
        if comp == "imag":
            plt.ylabel(r"attenuation constant")
        else:
            plt.ylabel(r"phase constant")
        plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.0)

    def plot_e_field(
        self,
        w: complex,
        dir: str,
        alpha: tuple[str, int, int],
        x_max: float = 0.25,
        y_max: float = 0.25,
    ):
        """Plot the electric field distribution in the cross section.

        Args:
            w: A complex indicating the angular frequency
            alpha: A tuple (pol, n, m) where pol is 'M' for TM-like mode or
                'E' for TE-like mode, n is the order of the mode, and m is
                the number of modes in the order and the polarization.
            dir: "h" (horizontal polarization) or "v" (vertical polarization)
            x_max: A float indicating the maximum x coordinate in the figure
            y_max: A float indicating the maximum y coordinate in the figure
        """
        xs = np.linspace(-x_max, x_max, 129)
        ys = np.linspace(-y_max, y_max, 129)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        h = self.beta(w, alpha)
        coef = self.coef(h, w, alpha)
        E = np.array(
            [[self.e_field(x, y, w, dir, alpha, h, coef) for y in ys] for x in xs]
        )
        Ex = E[:, :, 0]
        Ey = E[:, :, 1]
        Ez = E[:, :, 2]
        Es = np.sqrt(np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2)
        E_max_abs = Es.max()
        Es /= E_max_abs
        if dir == "h":
            Ex_on_x = Ex[:, 64]
            Ex_max = Ex_on_x[np.abs(Ex_on_x).argmax()]
            E_norm = Ex_max.conjugate() / abs(Ex_max) / E_max_abs
        else:
            Ey_on_y = Ey[64]
            Ey_max = Ey_on_y[np.abs(Ey_on_y).argmax()]
            E_norm = Ey_max.conjugate() / abs(Ey_max) / E_max_abs
        Ex = (Ex * E_norm).real
        Ey = (Ey * E_norm).real
        # Ez = (Ez * E_norm).real
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect="equal")
        pc = ax.pcolormesh(X, Y, Es, shading="auto", rasterized=True)
        # circle = Circle((0.0, 0.0), self.r, fill=False, ls='solid', color='w')
        # ax.add_patch(circle)
        ax.quiver(
            X[2::5, 2::5],
            Y[2::5, 2::5],
            Ex[2::5, 2::5],
            Ey[2::5, 2::5],
            scale=16.0,
            width=0.006,
            color="k",
            pivot="middle",
        )
        ax.set_xlim(-x_max, x_max)
        ax.set_ylim(-y_max, y_max)
        ax.set_xlabel(r"$x\ [\mu\mathrm{m}]$")  # , size=20)
        ax.set_ylabel(r"$y\ [\mu\mathrm{m}]$")  # , size=20)
        # plt.tick_params(labelsize=18)
        plt.colorbar(pc, pad=0.02)
        # cbar.ax.tick_params(labelsize=14)
        plt.tight_layout()
        plt.show()

    def plot_h_field(self, w, dir, alpha, x_max=0.25, y_max=0.25):
        """Plot the magnetic field distribution in the cross section.

        Args:
            w: A complex indicating the angular frequency
            dir: "h" (horizontal polarization) or "v" (vertical polarization)
            alpha: A tuple (pol, n, m) where pol is 'M' for TM-like mode or
                'E' for TE-like mode, n is the order of the mode, and m is
                the number of modes in the order and the polarization.
            x_max: A float indicating the maximum x coordinate in the figure
            y_max: A float indicating the maximum y coordinate in the figure
        """
        xs = np.linspace(-x_max, x_max, 129)
        ys = np.linspace(-y_max, y_max, 129)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        h = self.beta(w, alpha)
        coef = self.coef(h, w, alpha)
        H = np.array(
            [[self.h_field(x, y, w, dir, alpha, h, coef) for y in ys] for x in xs]
        )
        Hx = H[:, :, 0]
        Hy = H[:, :, 1]
        Hz = H[:, :, 2]
        Hs = np.sqrt(np.abs(Hx) ** 2 + np.abs(Hy) ** 2 + np.abs(Hz) ** 2)
        H_max_abs = Hs.max()
        Hs /= H_max_abs
        if dir == "h":
            Hy_on_x = Hy[:, 64]
            Hy_max = Hy_on_x[np.abs(Hy_on_x).argmax()]
            H_norm = Hy_max.conjugate() / abs(Hy_max) / H_max_abs
        else:
            Hx_on_y = Hx[64]
            Hx_max = Hx_on_y[np.abs(Hx_on_y).argmax()]
            H_norm = Hx_max.conjugate() / abs(Hx_max) / H_max_abs
        Hx = (Hx * H_norm).real
        Hy = (Hy * H_norm).real
        # Hz = (Hz * H_norm).real
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect="equal")
        pc = ax.pcolormesh(X, Y, Hs, shading="auto", rasterized=True)
        # circle = Circle((0.0, 0.0), self.r, fill=False, ls='solid', color='w')
        # ax.add_patch(circle)
        ax.quiver(
            X[2::5, 2::5],
            Y[2::5, 2::5],
            Hx[2::5, 2::5],
            Hy[2::5, 2::5],
            scale=16.0,
            width=0.006,
            color="k",
            pivot="middle",
        )
        ax.set_xlim(-x_max, x_max)
        ax.set_ylim(-y_max, y_max)
        ax.set_xlabel(r"$x\ [\mu\mathrm{m}]$")  # , size=20)
        ax.set_ylabel(r"$y\ [\mu\mathrm{m}]$")  # , size=20)
        # plt.tick_params(labelsize=18)
        plt.colorbar(pc, pad=0.02)
        # cbar.ax.tick_params(labelsize=14)
        plt.tight_layout()
        plt.show()

    def plot_e_field_on_x_axis(self, w, dir, alpha, comp, x_max=0.3, nx=128):
        """Plot a component of the electric field on the x axis

        Args:
            w: A complex indicating the angular frequency
            dir: "h" (horizontal polarization) or "v" (vertical polarization)
            alpha: A tuple (pol, n, m) where pol is 'M' for TM-like mode or
                'E' for TE-like mode, n is the order of the mode, and m is
                the number of modes in the order and the polarization.
            comp: "x", "y" or "z" indicating the component to be drawn.
            x_max: A float indicating the maximum x coordinate in the figure
            nx: An integer indicating the number of calculational points
                (default: 128)
        """
        xs = np.linspace(-x_max, x_max, nx + 1)
        ys = np.linspace(-x_max, x_max, nx + 1)
        _, Y = np.meshgrid(xs, ys, indexing="ij")
        h = self.beta(w, alpha)
        coef = self.coef(h, w, alpha)
        E = np.array(
            [[self.e_field(x, y, w, dir, alpha, h, coef) for y in ys] for x in xs]
        )
        Ex = E[:, :, 0]
        Ey = E[:, :, 1]
        Ez = E[:, :, 2]
        Es = np.sqrt(np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2)
        E_max_abs = Es.max()
        if dir == "h":
            Ex_on_x = Ex[:, 64]
            Ex_max = Ex_on_x[np.abs(Ex_on_x).argmax()]
            E_norm = Ex_max.conjugate() / abs(Ex_max) / E_max_abs
        else:
            Ey_on_y = Ey[64]
            Ey_max = Ey_on_y[np.abs(Ey_on_y).argmax()]
            E_norm = Ey_max.conjugate() / abs(Ey_max) / E_max_abs
        Ex = (Ex * E_norm)[:, nx // 2].real
        Ey = (Ey * E_norm)[:, nx // 2].real
        Ez = (Ez * E_norm)[:, nx // 2].real
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if comp == "x":
            ax.plot(xs, Ex, "o-")
            ax.set_ylabel(r"$E_x$", size=20)
        elif comp == "y":
            ax.plot(xs, Ey, "o-")
            ax.set_ylabel(r"$E_y$", size=20)
        elif comp == "z":
            ax.plot(xs, Ez, "o-")
            ax.set_ylabel(r"$E_z$", size=20)
        else:
            raise ValueError("comp must be x, y or z")
        ax.set_xlim(-x_max, x_max)
        ax.set_xlabel(r"$x\ [\mu\mathrm{m}]$", size=20)
        plt.tick_params(labelsize=18)
        plt.tight_layout()
        plt.show()

    def plot_h_field_on_x_axis(self, w, dir, alpha, comp, x_max=0.3, nx=128):
        """Plot a component of the magnetic field on the x axis

        Args:
            w: A complex indicating the angular frequency
            dir: "h" (horizontal polarization) or "v" (vertical polarization)
            alpha: A tuple (pol, n, m) where pol is 'M' for TM-like mode or
                'E' for TE-like mode, n is the order of the mode, and m is
                the number of modes in the order and the polarization.
            comp: "x", "y" or "z" indicating the component to be drawn.
            x_max: A float indicating the maximum x coordinate in the figure
            nx: An integer indicating the number of calculational points
                (default: 128)
        """
        xs = np.linspace(-x_max, x_max, nx + 1)
        ys = np.linspace(-x_max, x_max, nx + 1)
        _, Y = np.meshgrid(xs, ys, indexing="ij")
        h = self.beta(w, alpha)
        coef = self.coef(h, w, alpha)
        H = np.array(
            [[self.h_field(x, y, w, dir, alpha, h, coef) for y in ys] for x in xs]
        )
        Hx = H[:, :, 0]
        Hy = H[:, :, 1]
        Hz = H[:, :, 2]
        Hs = np.sqrt(np.abs(Hx) ** 2 + np.abs(Hy) ** 2 + np.abs(Hz) ** 2)
        H_max_abs = Hs.max()
        if dir == "h":
            Hy_on_x = Hy[:, 64]
            Hy_max = Hy_on_x[np.abs(Hy_on_x).argmax()]
            H_norm = Hy_max.conjugate() / abs(Hy_max) / H_max_abs
        else:
            Hx_on_y = Hx[64]
            Hx_max = Hx_on_y[np.abs(Hx_on_y).argmax()]
            H_norm = Hx_max.conjugate() / abs(Hx_max) / H_max_abs
        Hx = (Hx * H_norm)[:, nx // 2].real
        Hy = (Hy * H_norm)[:, nx // 2].real
        Hz = (Hz * H_norm)[:, nx // 2].real
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if comp == "x":
            ax.plot(xs, Hx, "o-")
            ax.set_ylabel(r"$H_x$", size=20)
        elif comp == "y":
            ax.plot(xs, Hy, "o-")
            ax.set_ylabel(r"$H_y$", size=20)
        elif comp == "z":
            ax.plot(xs, Hz, "o-")
            ax.set_ylabel(r"$H_z$", size=20)
        else:
            raise ValueError("comp must be x, y or z")
        ax.set_xlim(-x_max, x_max)
        ax.set_xlabel(r"$x\ [\mu\mathrm{m}]$", size=20)
        plt.tick_params(labelsize=18)
        plt.tight_layout()
        plt.show()


class Database:
    """The interface with the database of propagation constants."""

    dirname = os.path.join(os.path.expanduser("~"), ".pymwm")
    filename = os.path.join(dirname, "pymwm_data.h5")
    catalog_columns = OrderedDict(
        (
            ("sn", int),
            ("shape", str),
            ("size", float),
            ("size2", float),
            ("core", str),
            ("clad", str),
            ("wl_max", float),
            ("wl_min", float),
            ("wl_imag", float),
            ("dw", float),
            ("num_n", int),
            ("num_m", int),
            ("EM", str),
            ("n", int),
            ("m", int),
        )
    )
    data_columns = OrderedDict(
        (("conv", bool), ("beta_real", float), ("beta_imag", float))
    )

    def __init__(self, key: Optional[dict] = None) -> None:
        if key is None:
            self.cond = "shape=='None'"
        else:
            self.shape = key["shape"]
            self.size = key["size"]
            self.size2 = key["size2"]
            self.core = key["core"]
            self.clad = key["clad"]
            self.wl_max = key["wl_max"]
            self.wl_min = key["wl_min"]
            self.wl_imag = key["wl_imag"]
            self.dw = key["dw"]
            self.num_n = key["num_n"]
            self.num_m = key["num_m"]
            self.num_all = key["num_all"]
            cond = ""
            for col in list(self.catalog_columns.keys())[1:-3]:
                cond += f"{col} == @self.{col} & "
            self.cond = cond.rstrip("& ")
            ind_w_min = int(np.floor(2 * np.pi / self.wl_max / self.dw))
            ind_w_max = int(np.ceil(2 * np.pi / self.wl_min / self.dw))
            ind_w_imag = int(np.ceil(2 * np.pi / self.wl_imag / self.dw))
            self.ws = np.arange(ind_w_min, ind_w_max + 1) * self.dw
            self.wis = -np.arange(ind_w_imag + 1) * self.dw
        self.sn = self.get_sn()

    def load_catalog(self) -> DataFrame:
        df: DataFrame = pd.read_hdf(self.filename, "catalog")
        return df

    def get_sn(self) -> int:
        if not os.path.exists(self.filename):
            if not os.path.exists(self.dirname):
                os.mkdir(self.dirname)
            catalog = pd.DataFrame(columns=self.catalog_columns.keys())
            catalog.to_hdf(self.filename, "catalog", complevel=9, complib="blosc")
            return 0
        catalog = self.load_catalog()
        if len(catalog.index) == 0:
            return 0
        sns = catalog.query(self.cond)["sn"]
        if len(sns):
            if len(sns) != self.num_all:
                print(sns)
                print(catalog[self.cond])
                print(len(sns), self.num_all)
                raise Exception("Database is broken.")
            sn: int = min(sns)
        else:
            sn = max(catalog["sn"]) + 1
        return sn

    @staticmethod
    def set_columns_dtype(df: DataFrame, columns: dict):
        """Set data type of each column in the DataFrame."""
        for key, val in columns.items():
            df[key] = df[key].astype(val)

    def load(self) -> tuple[dict, dict]:
        num_wr = len(self.ws)
        num_wi = len(self.wis)
        betas = dict()
        convs = dict()
        sns = range(self.sn, self.sn + self.num_all)
        catalog = pd.read_hdf(self.filename, "catalog")
        indices = [catalog[catalog["sn"] == sn].index[0] for sn in sns]
        #  If there is no data for sn, IndexError should be raised
        #  in the following expression.
        for i, sn in zip(indices, sns):
            em = catalog.loc[i, "EM"]
            n = catalog.loc[i, "n"]
            m = catalog.loc[i, "m"]
            data = pd.read_hdf(self.filename, f"sn_{sn}")
            conv = data["conv"]
            beta_real = data["beta_real"]
            beta_imag = data["beta_imag"]
            convs[(em, n, m)] = conv.values.reshape(num_wr, num_wi)
            beta = np.zeros_like(beta_real.values, dtype=np.complex128)
            beta.real = beta_real.values
            beta.imag = beta_imag.values
            betas[(em, n, m)] = beta.reshape(num_wr, num_wi)
        return betas, convs

    def save(self, betas: dict, convs: dict):
        catalog = pd.read_hdf(self.filename, "catalog")
        indices = catalog.query(self.cond).index
        sns = catalog.query(self.cond)["sn"]
        with pd.HDFStore(self.filename) as store:
            for i, sn in zip(indices, sns):
                catalog = catalog.drop(i)
                del store[f"sn_{sn}"]
        dfs = {}
        sn = self.sn
        for EM, n, m in sorted(convs.keys()):
            se = pd.Series(
                [
                    sn,
                    self.shape,
                    self.size,
                    self.size2,
                    self.core,
                    self.clad,
                    self.wl_max,
                    self.wl_min,
                    self.wl_imag,
                    self.dw,
                    self.num_n,
                    self.num_m,
                    EM,
                    n,
                    m,
                ],
                index=self.catalog_columns.keys(),
            )
            catalog = catalog.append(se, ignore_index=True)
            conv = convs[(EM, n, m)].ravel()
            beta = betas[(EM, n, m)].ravel()
            df = pd.DataFrame(
                {"conv": conv, "beta_real": beta.real, "beta_imag": beta.imag},
                columns=self.data_columns.keys(),
            )
            self.set_columns_dtype(df, self.data_columns)
            dfs[sn] = df
            sn += 1
        self.set_columns_dtype(catalog, self.catalog_columns)
        with pd.HDFStore(self.filename, complevel=9, complib="blosc") as store:
            for sn, df in dfs.items():
                store[f"sn_{sn}"] = df
            store["catalog"] = catalog

    def compress(self):
        os.system(
            "ptrepack --chunkshape=auto --propindexes --complevel=9 "
            + "--complib=blosc {0}: {0}.new:".format(self.filename)
        )
        os.system("mv {0}.new {0}".format(self.filename))

    @classmethod
    def import_data(cls, data_file: str):
        with pd.HDFStore(data_file, "r") as data:
            catalog_from = data["catalog"]
            sns = catalog_from["sn"]
            data_dict = {sn: data["sn_{}".format(sn)] for sn in sns}
        with pd.HDFStore(cls.filename, complevel=9, complib="blosc") as store:
            catalog = store["catalog"]
            sn_new = max(catalog["sn"]) + 1
            for sn in sns:
                se = catalog_from[catalog_from["sn"] == sn].iloc[0].copy()
                cond = catalog["shape"] == se["shape"]
                for col in list(cls.catalog_columns.keys())[2:]:
                    cond &= catalog[col] == se[col]
                if len(catalog[cond].index) == 0:
                    se["sn"] = sn_new
                    catalog = catalog.append(se, ignore_index=True)
                    print(catalog[catalog["sn"] == sn_new])
                    store.append("sn_{}".format(sn_new), data_dict[sn])
                    sn_new += 1
            cls.set_columns_dtype(catalog, cls.catalog_columns)
            store["catalog"] = catalog

    def delete(self, sns: list):
        with pd.HDFStore(self.filename, complevel=9, complib="blosc") as store:
            catalog = store["catalog"]
            indices = [catalog[catalog["sn"] == sn].index[0] for sn in sns]
            for i, sn in zip(indices, sns):
                catalog.drop(i, inplace=True)
                del store[f"sn_{sn}"]
            store["catalog"] = catalog
        self.sn = self.get_sn()

    def delete_current(self):
        with pd.HDFStore(self.filename, complevel=9, complib="blosc") as store:
            catalog = store["catalog"]
            sns = range(self.sn, self.sn + self.num_all)
            indices = [catalog[catalog["sn"] == sn].index[0] for sn in sns]
            for i, sn in zip(indices, sns):
                catalog.drop(i, inplace=True)
                del store[f"sn_{sn}"]
            store["catalog"] = catalog
        self.sn = self.get_sn()

    def interpolation(self, betas: np.ndarray, convs: np.ndarray, bounds: dict) -> dict:
        from scipy.interpolate import RectBivariateSpline

        wl_max = bounds["wl_max"]
        wl_min = bounds["wl_min"]
        wl_imag = bounds["wl_imag"]
        beta_imag_max = bounds.get("beta_imag_max", None)
        wr_min = 2 * np.pi / wl_max
        wr_max = 2 * np.pi / wl_min
        wi_min = -2 * np.pi / wl_imag
        num_n = self.num_n
        num_m = self.num_m
        ws = self.ws
        wis = self.wis[::-1]
        i_min = np.searchsorted(ws, [wr_min], side="right")[0] - 1
        i_max = np.searchsorted(ws, [wr_max])[0]
        j_min = np.searchsorted(wis, [wi_min], side="right")[0] - 1
        if i_min == -1 or i_max == len(self.ws) or j_min == -1:
            raise ValueError(
                "exceed data bounds: "
                + "i_min={} i_max={} j_min={} len(ws)={}".format(
                    i_min, i_max, j_min, len(self.ws)
                )
            )
        beta_funcs = {}
        for pol in ["M", "E"]:
            for n in range(num_n):
                for m in range(1, num_m + 1):
                    alpha = (pol, n, m)
                    conv = convs[alpha][:, ::-1]
                    if np.all(conv[i_min : i_max + 1, j_min:]):
                        data = betas[alpha][:, ::-1]
                        if beta_imag_max is not None:
                            imag_max = data[i_min : i_max + 1, j_min:].imag.max()
                            if imag_max > beta_imag_max:
                                continue
                        beta_funcs[(alpha, "real")] = RectBivariateSpline(
                            ws[i_min : i_max + 1],
                            wis[j_min:],
                            data.real[i_min : i_max + 1, j_min:],
                            kx=3,
                            ky=3,
                        )
                        beta_funcs[(alpha, "imag")] = RectBivariateSpline(
                            ws[i_min : i_max + 1],
                            wis[j_min:],
                            data.imag[i_min : i_max + 1, j_min:],
                            kx=3,
                            ky=3,
                        )
        return beta_funcs
