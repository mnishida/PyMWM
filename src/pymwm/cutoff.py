from __future__ import annotations

import os

import numpy as np
import pandas as pd
import scipy.optimize as so
import scipy.special as sp

from pymwm.utils.cutoff_utils import f_fp_fpp_cython


class Cutoff:
    """A callable class that calculates the values of u at cutoff frequencies for coaxial waveguides made of PEC

    Attributes:
        num_n (int): The number of orders of modes.
        num_m (int): The number of modes in each order and polarization.
        r_ratios (list[float]): A list of ratio between inner and outer radii.
        samples (DataFrame): Cutoff values of u at r_ratios.
    """

    dirname = os.path.join(os.path.expanduser("~"), ".pymwm")
    filename = os.path.join(dirname, "cutoff.h5")

    def __init__(self, num_n: int, num_m: int) -> None:
        """Init Cutoff class."""
        self.num_n, self.num_m = num_n, num_m
        self.r_ratios = 0.001 * np.arange(1000)
        if os.path.exists(self.filename):
            samples = pd.read_hdf(self.filename)
            if self.num_n > samples["n"].max() + 1 or self.num_m >= samples["m"].max():
                self.samples = self.cutoffs()
                self.samples.to_hdf(self.filename, "cutoff")
            else:
                self.samples = samples[
                    (samples["n"] < num_n)
                    & (
                        (samples["pol"] == "E") & (samples["m"] <= num_m)
                        | (samples["pol"] == "M") & (samples["m"] <= num_m + 1)
                    )
                ].reset_index(drop=True)
        if not os.path.exists(self.filename):
            if not os.path.exists(self.dirname):
                print("Folder Not Found.")
                os.mkdir(self.dirname)
            print("File Not Found.")
            self.samples = self.cutoffs()
            self.samples.to_hdf(self.filename, "cutoff")

    def __call__(self, alpha: tuple, r_ratio: float) -> float:
        """Return the cutoff value of u

        Args:
            alpha (tuple[pol (str), n (int), m (int)]):
                pol: 'E' or 'M' indicating the polarization.
                n: A integer indicating the order of the mode.
                m: A integer indicating the ordinal of the mode in the same
                    order.
            r_ratio (float): The ratio between inner and outer radii.

        Returns:
            float: The value of u at cutoff frequency.
        """
        df = self.samples
        df = df[(df["pol"] == alpha[0]) & (df["n"] == alpha[1]) & (df["m"] == alpha[2])]
        x = df["rr"].to_numpy()
        y = df["val"].to_numpy()
        i: int = np.where(x <= r_ratio)[0][-1]
        a: float = r_ratio - x[i]
        b: float = x[i + 1] - r_ratio
        val: float = (b * y[i] + a * y[i + 1]) / (a + b)
        return val

    def PEC(
        self, u: float, r_ratio: float, n: int, pol: str
    ) -> tuple[float, float, float]:
        x = u * r_ratio

        def f_fp_fpp_fppp(func, z):
            f = func(n, z)
            fp = -func(n + 1, z) + n / z * f
            fpp = -fp / z - (1 - n ** 2 / z ** 2) * f
            fppp = -fpp / z - fp + (n ** 2 + 1) * fp / z ** 2 - 2 * n ** 2 / z ** 3 * f
            return f, fp, fpp, fppp

        jx, jpx, jppx, jpppx = f_fp_fpp_fppp(sp.jv, x)
        yx, ypx, yppx, ypppx = f_fp_fpp_fppp(sp.yv, x)
        ju, jpu, jppu, jpppu = f_fp_fpp_fppp(sp.jv, u)
        yu, ypu, yppu, ypppu = f_fp_fpp_fppp(sp.yv, u)
        if pol == "E":
            f = jpx * ypu - ypx * jpu
            fp = r_ratio * jppx * ypu + jpx * yppu - r_ratio * yppx * jpu - ypx * jppu
            fpp = (
                r_ratio ** 2 * jpppx * ypu
                + 2 * r_ratio * jppx * yppu
                + jpx * ypppu
                - r_ratio ** 2 * ypppx * jpu
                - 2 * r_ratio * yppx * jppu
                - ypx * jpppu
            )
        else:
            f = jx * yu - yx * ju
            fp = r_ratio * jpx * yu + jx * ypu - r_ratio * ypx * ju - yx * jpu
            fpp = (
                r_ratio ** 2 * jppx * yu
                + 2 * r_ratio * jpx * ypu
                + jx * yppu
                - r_ratio ** 2 * yppx * ju
                - 2 * r_ratio * ypx * jpu
                - yx * jppu
            )
        return f, fp, fpp

    def cutoffs_numpy(self) -> pd.DataFrame:
        import ray

        if not ray.is_initialized():
            ray.init()
        rrs_id = ray.put(self.r_ratios)
        pec_id = ray.put(self.PEC)

        @ray.remote
        def func(alpha, kini, rrs, pec):
            pol, n, m = alpha
            drr = 0.1 * (rrs[1] - rrs[0])
            x0 = x1 = kini
            z = []
            for rr in rrs:
                z.append(x1)
                for i in range(1, 11):
                    sol = so.root_scalar(
                        pec,
                        x0=2 * x1 - x0,
                        fprime=True,
                        fprime2=True,
                        method="halley",
                        args=(rr + i * drr, n, pol),
                    )
                    x0 = x1
                    x1 = sol.root
            return z

        num_m = self.num_m
        args = []
        for n in range(self.num_n):
            for (pol, m_end) in [("M", num_m + 2), ("E", num_m + 1)]:
                if pol == "E":
                    kinis = sp.jnp_zeros(n, m_end - 1)
                else:
                    kinis = sp.jn_zeros(n, m_end - 1)
                for m in range(1, m_end):
                    kini = kinis[m - 1]
                    args.append(((pol, n, m), kini))

        result_ids = [func.remote(alpha, kini, rrs_id, pec_id) for alpha, kini in args]
        results = ray.get(result_ids)
        if ray.is_initialized():
            ray.shutdown()
        df = pd.DataFrame()
        num_rr = len(self.r_ratios)
        for i in range(len(args)):
            (pol, n, m), _ = args[i]
            z = results[i]
            df1 = pd.DataFrame()
            df1["pol"] = np.full(num_rr, pol)
            df1["n"] = n
            df1["m"] = m
            df1["irr"] = np.arange(num_rr)
            df1["rr"] = self.r_ratios
            df1["val"] = z
            df = pd.concat([df, df1], ignore_index=True)
        return df

    def cutoffs(self) -> pd.DataFrame:
        import ray

        if not ray.is_initialized():
            ray.init()
        rrs_id = ray.put(self.r_ratios)

        @ray.remote
        def func(alpha, kini, rrs):
            pol, n, m = alpha
            drr = 0.1 * (rrs[1] - rrs[0])
            x0 = x1 = kini
            z = []
            for rr in rrs:
                z.append(x1)
                for i in range(1, 11):
                    sol = so.root_scalar(
                        f_fp_fpp_cython,
                        x0=2 * x1 - x0,
                        fprime=True,
                        fprime2=True,
                        method="halley",
                        args=(rr + i * drr, n, pol),
                    )
                    x0 = x1
                    x1 = sol.root
            return z

        num_m = self.num_m
        args = []
        for n in range(self.num_n):
            for (pol, m_end) in [("M", num_m + 2), ("E", num_m + 1)]:
                if pol == "E":
                    kinis = sp.jnp_zeros(n, m_end - 1)
                else:
                    kinis = sp.jn_zeros(n, m_end - 1)
                for m in range(1, m_end):
                    kini = kinis[m - 1]
                    args.append(((pol, n, m), kini))

        result_ids = [func.remote(alpha, kini, rrs_id) for alpha, kini in args]
        results = ray.get(result_ids)
        if ray.is_initialized():
            ray.shutdown()
        df = pd.DataFrame()
        num_rr = len(self.r_ratios)
        for i in range(len(args)):
            (pol, n, m), _ = args[i]
            z = results[i]
            df1 = pd.DataFrame()
            df1["pol"] = np.full(num_rr, pol)
            df1["n"] = n
            df1["m"] = m
            df1["irr"] = np.arange(num_rr)
            df1["rr"] = self.r_ratios
            df1["val"] = z
            df = pd.concat([df, df1], ignore_index=True)
        return df
