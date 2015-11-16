#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
from nose.tools import assert_equal, assert_true


def test_palik():
    import numpy as np
    from pymwm.material import Material
    unit = 0.5
    gold = Material({'model': 'gold_palik', 'unit': unit})
    silver = Material({'model': 'silver_palik', 'unit': unit})
    copper = Material({'model': 'copper_palik', 'unit': unit})

    palik = gold._palik
    # import matplotlib.pyplot as plt
    # plt.plot(palik.ws, palik.ns ** 2 - palik.ks ** 2, "o")
    # plt.plot(palik.ws, 2 * palik.ns * palik.ks, "v")
    # ws = np.linspace(palik.ws.min(), palik.ws.max(), 10000)
    # plt.plot(ws, np.vectorize(gold)(ws).real)
    # plt.plot(ws, np.vectorize(gold)(ws).imag)
    # plt.show()

    assert_true(np.allclose(
        np.vectorize(gold)(palik.ws),
        palik.ns ** 2 - palik.ks ** 2 + 2j * palik.ns * palik.ks))
    palik = silver._palik
    assert_true(np.allclose(
        np.vectorize(silver)(palik.ws),
        palik.ns ** 2 - palik.ks ** 2 + 2j * palik.ns * palik.ks))
    palik = copper._palik
    assert_true(np.allclose(
        np.vectorize(copper)(palik.ws),
        palik.ns ** 2 - palik.ks ** 2 + 2j * palik.ns * palik.ks))
