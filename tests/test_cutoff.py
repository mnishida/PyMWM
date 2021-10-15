import cmath

import numpy as np
import numpy.testing as npt
import pandas as pd
import scipy.special
from numpy.testing._private.utils import assert_equal

from pymwm.cutoff import Cutoff


def test_cutoffs():
    co = Cutoff(2, 2)
    num_rr = len(co.r_ratios)
    assert co.num_n == 2
    assert co.num_m == 2
    assert len(co.samples) == co.num_n * (2 * co.num_m + 1) * num_rr
    co = Cutoff(2, 1)
    assert co.num_m == 1
    assert len(co.samples) == co.num_n * (2 * co.num_m + 1) * num_rr


def test_cutoffs_TE_r1_0():
    co = Cutoff(3, 3)
    num_rr = len(co.r_ratios)
    assert co.num_n == 3
    assert co.num_m == 3
    assert len(co.samples) == co.num_n * (2 * co.num_m + 1) * num_rr
    for n in range(3):
        bessels = scipy.special.jnp_zeros(n, 3)
        for m in range(1, 4):
            s = co.samples.query(f"pol=='E' and n=={n} and m=={m} and irr==0").iloc[0]
            print(s["val"], bessels[m - 1])
            assert s["val"] == bessels[m - 1]


def test_cutoffs_TM_r1_0():
    co = Cutoff(3, 3)
    s = co.samples.query("pol=='M' and n==0 and m==1 and irr==0").iloc[0]
    assert s["val"] == 0.0
    bessels = scipy.special.jn_zeros(0, 4)
    for m in range(2, 5):
        s = co.samples.query(f"pol=='M' and n==0 and m=={m} and irr==0").iloc[0]
        print(s["val"], bessels[m - 2])
        assert s["val"] == bessels[m - 2]
    for n in range(1, 3):
        bessels = scipy.special.jn_zeros(n, 4)
        for m in range(1, 5):
            s = co.samples.query(f"pol=='M' and n=={n} and m=={m} and irr==0").iloc[0]
            print(s["val"], bessels[m - 1])
            assert s["val"] == bessels[m - 1]


def test_cutoffs_TE():
    co = Cutoff(2, 3)
    num_rr = len(co.r_ratios)
    assert co.num_n == 2
    assert co.num_m == 3
    assert len(co.samples) == co.num_n * (2 * co.num_m + 1) * num_rr
    a = 0.1
    c = 0.5
    r_ratio = a / c
    k0 = 0.3
    eps = 1
    kc = co(("E", 1, 2), r_ratio) / c
    beta = (1 + 1j) * cmath.sqrt(-0.5j * (eps * k0 ** 2 - kc ** 2))
    val = beta ** 2
    print(val)
    npt.assert_allclose(val, -98.35032331623853 + 0j, rtol=1e-6)


def test_cutoffs_cython():
    co = Cutoff(3, 3)
    num_rrs = len(co.r_ratios)
    df1 = co.cutoffs()
    df2 = co.cutoffs_numpy()
    for pol, m_end in [("M", 4), ("E", 4)]:
        for n in range(3):
            for m in range(1, m_end):
                for irr in range(num_rrs):
                    val1 = df1[
                        (df1["pol"] == pol)
                        & (df1["n"] == n)
                        & (df1["m"] == m)
                        & (df1["irr"] == irr)
                    ]["val"].iloc[0]
                    val2 = df2[
                        (df2["pol"] == pol)
                        & (df2["n"] == n)
                        & (df2["m"] == m)
                        & (df2["irr"] == irr)
                    ]["val"].iloc[0]
                    try:
                        npt.assert_almost_equal(val1, val2)
                    except Exception as e:
                        print(pol, n, m, irr, num_rrs, val1, val2)
                        raise e


def test_cutoffs_samples(num_regression):
    df = Cutoff(16, 8).samples
    d = {}
    for pol, m_end in [("M", 10), ("E", 9)]:
        for n in range(16):
            for m in range(1, m_end):
                df1 = df[(df["pol"] == pol) & (df["n"] == n) & (df["m"] == m)]
                d[f"{pol}_{n}_{m}_rr"] = df1["rr"].to_numpy()
                d[f"{pol}_{n}_{m}_val"] = df1["val"].to_numpy()
    num_regression.check(d)
