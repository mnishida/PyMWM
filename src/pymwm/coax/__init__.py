from __future__ import annotations

import cmath
from typing import Optional

import numpy as np
import ray

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
            smp = Samples(self.r, self.fill_params, self.clad_params, p_modes, self.ri)
            ray.shutdown()
            try:
                ray.init()
                p_modes_id = ray.put(p_modes)
                pool = ray.util.ActorPool(
                    SamplesForRay.remote(
                        self.r, self.fill_params, self.clad_params, p_modes_id, self.ri
                    )
                    for _ in range(num_n_0)
                )
                xs_success_list = list(
                    pool.map(lambda a, arg: a.task.remote(arg), range(num_n_0))
                )
            finally:
                ray.shutdown()
            betas, convs = smp.betas_convs(xs_success_list)
            smp.database.save(betas, convs)
        if im_factor != 1.0:
            self.clad.im_factor = im_factor
            self.clad_params["im_factor"] = im_factor
            smp = SamplesLowLoss(
                self.r, self.fill_params, self.clad_params, p_modes, self.ri
            )
            try:
                betas, convs = smp.database.load()
            except IndexError:
                self.clad.im_factor = im_factor
                self.clad_params["im_factor"] = im_factor
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
                            self.r,
                            self.fill_params,
                            self.clad_params,
                            p_modes_id,
                            self.ri,
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
        if pol == "M":
            chi = self.samples.co_list[n][m]
        else:
            chi = self.samples.co_list[n][self.num_m + m + 1]
        val = cmath.sqrt(self.fill(w_comp) * w_comp ** 2 - chi ** 2 / self.r ** 2)
        if abs(val.real) > abs(val.imag):
            if val.real < 0:
                val *= -1
        else:
            if val.imag < 0:
                val *= -1
        return val
