#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
from nose.tools import assert_equal, assert_true


def test_johnson_christy():
    import numpy as np
    from pymwm.material import Material
    gold = Material({'model': 'gold_jc'})
    silver = Material({'model': 'silver_jc'})
    copper = Material({'model': 'copper_jc'})

    jc = gold._johnson_christy
    # import matplotlib.pyplot as plt
    # plt.plot(jc.ws, jc.ns ** 2 - jc.ks ** 2, "o")
    # plt.plot(jc.ws, 2 * jc.ns * jc.ks, "v")
    # ws = np.linspace(jc.ws.min(), jc.ws.max(), 1000)
    # plt.plot(ws, np.vectorize(gold)(ws).real)
    # plt.plot(ws, np.vectorize(gold)(ws).imag)
    # plt.show()

    assert_true(np.allclose(
        np.vectorize(gold)(jc.ws),
        jc.ns ** 2 - jc.ks ** 2 + 2j * jc.ns * jc.ks))
    jc = silver._johnson_christy
    assert_true(np.allclose(
        np.vectorize(silver)(jc.ws),
        jc.ns ** 2 - jc.ks ** 2 + 2j * jc.ns * jc.ks))
    jc = copper._johnson_christy
    assert_true(np.allclose(
        np.vectorize(copper)(jc.ws),
        jc.ns ** 2 - jc.ks ** 2 + 2j * jc.ns * jc.ks))
