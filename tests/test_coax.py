#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.special as sp
import numpy.testing as npt
from pymwm.cutoff import Cutoff


def test_cutoffs():
    co = Cutoff(2, 2)
    z = co.cutoffs()
    assert len(z) == 100


def test_cutoffs_TE_r1_0():
    co = Cutoff(3, 3)
    cut = co.cutoffs()
    for n in range(3):
        bessels = sp.jnp_zeros(n, 3)
        for m in range(1, 4):
            assert cut[0][("E", n, m)] == bessels[m - 1]


def test_cutoffs_TM_r1_0():
    co = Cutoff(3, 3)
    cut = co.cutoffs()
    for n in range(3):
        bessels = sp.jn_zeros(n, 3)
        for m in range(1, 4):
            assert cut[0][("M", n, m)] == bessels[m - 1]


def test_cutoffs_TE():
    co = Cutoff(2, 3)
    a = 0.1
    c = 0.5
    r_ratio = a / c
    k0 = 0.3
    eps = 1
    kc = co.cutoffs()[int(r_ratio * len(co.cutoffs(2, 3)))][("E", 1, 2)] / c
    beta = (1 + 1j) * np.sqrt(-0.5j * (eps * k0 ** 2 - kc ** 2))
    val = beta ** 2
    npt.assert_allclose(val, -98.35032331623853 + 0j, rtol=1e-05)
