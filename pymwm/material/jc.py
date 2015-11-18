# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy.constants import eV, hbar, c
from scipy.interpolate import interp1d


class Johnson_Christy():

    """A class defining the dielectric function for noble metals according to
    P. B. Johnson and R. W. Christy, PRB 6, 4370 (1972).

    Attributes:
        ws: 1D array of floats indicating the angular frequencys.
        ns: 1D array of floats indicating the real part of RIs.
        ks: 1D array of floats indicating the imaginary part of RIs.
    """

    def __init__(self, metal, kind='cubic'):
        self.num = 2048
        dirname = os.path.dirname(__file__)
        dirname = os.path.join(dirname, "Johnson_Christy")
        filename = os.path.join(dirname, "{0}.npy".format(metal))
        data = np.load(filename)
        self.ws = data[:, 0] * eV / hbar * 1e-6 / c
        self.ns = data[:, 1]
        self.ks = data[:, 2]
        self.n_func = interp1d(self.ws, self.ns, kind=kind, copy=False,
                               assume_sorted=True)
        self.k_func = interp1d(self.ws, self.ks, kind=kind, copy=False,
                               assume_sorted=True)
        # self.ws = np.linspace(ws0[0], ws0[-1], self.num)
        # self.ns = self.n_func(self.ws)
        # self.ks = self.k_func(self.ws)

    def __call__(self, w):
        n = self.n_func(w) + 1j * self.k_func(w)
        return n ** 2
        # ind = np.searchsorted(self.ws, w)
        # if ind == 0 or ind == self.num:
        #     if w == self.ws[0]:
        #         n = self.ns[0]
        #         k = self.ks[0]
        #     else:
        #         raise ValueError("The frequency is out-of-range")
        # else:
        #     w0, w1 = self.ws[ind - 1: ind + 1]
        #     n0, n1 = self.ns[ind - 1: ind + 1]
        #     k0, k1 = self.ks[ind - 1: ind + 1]
        #     n = n0 + (n1 - n0) / (w1 - w0) * (w - w0)
        #     k = k0 + (k1 - k0) / (w1 - w0) * (w - w0)
        # return complex(n ** 2 - k ** 2, 2 * n * k)
