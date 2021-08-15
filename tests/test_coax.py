#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import numpy.testing as npt
import scipy.special as sp

from pymwm.cutoff import Cutoff


def test_cutoffs():
    co = Cutoff(2, 2)
    z = co.cutoffs()
    num_rr = len(co.r_ratios)
    num_pols = 2
    assert len(z) == co.num_n * co.num_m * num_pols * num_rr


def test_cutoffs_TE_r1_0():
    co = Cutoff(3, 3)
    cut = co.cutoffs()
    for n in range(3):
        bessels = sp.jnp_zeros(n, 3)
        for m in range(1, 4):
            s = cut.query(f"pol=='E' and n=={n} and m=={m} and irr==0").iloc[0]
            print(s["val"], bessels[m - 1])
            assert s["val"] == bessels[m - 1]


def test_cutoffs_TM_r1_0():
    co = Cutoff(3, 3)
    cut = co.cutoffs()
    for n in range(3):
        bessels = sp.jn_zeros(n, 3)
        for m in range(1, 4):
            s = cut.query(f"pol=='M' and n=={n} and m=={m} and irr==0").iloc[0]
            print(s["val"], bessels[m - 1])
            assert s["val"] == bessels[m - 1]


def test_cutoffs_TE():
    co = Cutoff(2, 3)
    a = 0.1
    c = 0.5
    r_ratio = a / c
    k0 = 0.3
    eps = 1
    kc = co(("E", 1, 2), r_ratio) / c
    beta = (1 + 1j) * np.sqrt(-0.5j * (eps * k0 ** 2 - kc ** 2))
    val = beta ** 2
    print(val)
    npt.assert_allclose(val, -98.35032331623853 + 0j, rtol=1e-6)
