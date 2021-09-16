from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.optimize as so
import scipy.special as sp

from pymwm.utils.cutoff_utils import f_cython, f_fprime_cython, fprime_cython


class Cutoff:

    dirname = os.path.join(os.path.expanduser("~"), ".pymwm")
    filename = os.path.join(dirname, "cutoff.h5")

    def __init__(self, num_n: int, num_m: int) -> None:
        """Init Cutoff class."""
        self.num_n, self.num_m = num_n, num_m
        self.r_ratios = 0.001 * np.arange(1000)
        if os.path.exists(self.filename):
            samples = pd.read_hdf(self.filename)
            if self.num_n > samples["n"].max() + 1 or self.num_m > samples["m"].max():
                self.samples = self.cutoffs()
                self.samples.to_hdf(self.filename, "cutoff")
            else:
                self.samples = samples[
                    (samples["n"] < num_n) & (samples["m"] <= num_m)
                ].reset_index()
        if not os.path.exists(self.filename):
            if not os.path.exists(self.dirname):
                print("Folder Not Found.")
                os.mkdir(self.dirname)
            print("File Not Found.")
            self.samples = self.cutoffs()
            self.samples.to_hdf(self.filename, "cutoff")

    def __call__(self, alpha: tuple, r_ratio: float) -> float:
        df = self.samples
        df = df[(df["pol"] == alpha[0]) & (df["n"] == alpha[1]) & (df["m"] == alpha[2])]
        print(df)
        x = df["rr"].to_numpy()
        y = df["val"].to_numpy()
        print(x)
        i: int = np.where(x < r_ratio)[0][-1]
        a: float = r_ratio - x[i]
        b: float = x[i + 1] - r_ratio
        val: float = (b * y[i] + a * y[i + 1]) / (a + b)
        return val

    def PEC(
        self, x: float, r_ratio: float, n: int, pol: str
    ) -> tuple[float, float, float]:
        v = x * r_ratio

        def f_fp_fpp_fppp(func, z):
            f = func(n, z)
            fp = -func(n + 1, z) + n / z * f
            fpp = -fp / z - (1 - n ** 2 / z ** 2) * f
            fppp = -fpp / z - fp + (n ** 2 + 1) * fp / z ** 2 - 2 * n ** 2 / z ** 3 * f
            return f, fp, fpp, fppp

        jv, jpv, jppv, jpppv = f_fp_fpp_fppp(sp.jv, v)
        yv, ypv, yppv, ypppv = f_fp_fpp_fppp(sp.yv, v)
        jx, jpx, jppx, jpppx = f_fp_fpp_fppp(sp.jv, x)
        yx, ypx, yppx, ypppx = f_fp_fpp_fppp(sp.yv, x)
        if pol == "E":
            f = jpv * ypx - ypv * jpx
            fp = r_ratio * jppv * ypx + jpv * yppx - r_ratio * yppv * jpx - ypv * jppx
            fpp = (
                r_ratio ** 2 * jpppv * ypx
                + 2 * r_ratio * jppv * yppx
                + jpv * ypppx
                - r_ratio ** 2 * ypppv * jpx
                - 2 * r_ratio * yppv * jppx
                - ypv * jpppx
            )
        else:
            f = jv * yx - yv * jx
            fp = r_ratio * jpv * yx + jv * ypx - r_ratio * ypv * jx - yv * jpx
            fpp = (
                r_ratio ** 2 * jppv * yx
                + 2 * r_ratio * jpv * ypx
                + jv * yppx
                - r_ratio ** 2 * yppv * jx
                - 2 * ypv * jpx
                - yv * jppx
            )
        return f, fp, fpp

    def cutoffs_old(self):
        num_n, num_m = self.num_n, self.num_m
        z = []
        data = {}
        for pol in ["E", "M"]:
            for n in range(num_n):
                if pol == "E":
                    kini = sp.jnp_zeros(n, num_m)
                else:
                    kini = sp.jn_zeros(n, num_m)
                for m in range(1, num_m + 1):
                    data[(pol, n, m)] = kini[m - 1]
        z.append(data)
        for r_ratio in self.r_ratios[1:]:
            data = {}
            for pol in ["E", "M"]:
                for n in range(num_n):
                    for m in range(1, num_m + 1):
                        if r_ratio == self.r_ratios[1]:
                            ini_val = z[-1][(pol, n, m)]
                        else:
                            ini_val = z[-1][(pol, n, m)] * 2 - z[-2][(pol, n, m)]
                        sol = so.root_scalar(
                            self.PEC,
                            x0=ini_val,
                            fprime=True,
                            fprime2=True,
                            args=(r_ratio, n, pol),
                        )
                        kini = sol.root
                        data[(pol, n, m)] = kini
            z.append(data)
        df = pd.DataFrame()
        for i, (r_ratio, sample) in enumerate(zip(self.r_ratios, z)):
            for pol in ["E", "M"]:
                for n in range(self.num_n):
                    for m in range(1, self.num_m + 1):
                        df = df.append(
                            {
                                "irr": i,
                                "rr": r_ratio,
                                "pol": pol,
                                "n": n,
                                "m": m,
                                "val": sample[(pol, n, m)],
                            },
                            ignore_index=True,
                        )
        return df.astype({"irr": int, "n": int, "m": int})

    def cutoffs(self) -> pd.DataFrame:
        import ray

        if not ray.is_initialized():
            ray.init()
        rrs_id = ray.put(self.r_ratios)

        @ray.remote
        def func(alpha, kini, rrs):
            pol, n, m = alpha
            z = [kini]
            sol = so.root_scalar(
                f_fprime_cython,
                x0=kini,
                fprime=True,
                method="newton",
                args=(rrs[1], n, pol),
            )
            z = [kini, sol.root]
            for r_ratio in rrs[2:]:
                ini_val = z[-1] * 2 - z[-2]
                sol = so.root_scalar(
                    f_fprime_cython,
                    x0=ini_val,
                    fprime=True,
                    method="newton",
                    args=(r_ratio, n, pol),
                )
                z.append(sol.root)
            return z

        num_m = self.num_m
        args = []
        for n in range(self.num_n):
            for pol in ["M", "E"]:
                if pol == "E":
                    kinis = sp.jnp_zeros(n, num_m)
                else:
                    kinis = sp.jn_zeros(n, num_m)
                for m in range(1, num_m + 1):
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
