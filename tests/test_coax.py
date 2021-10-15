from __future__ import annotations

import numpy as np
import numpy.testing as npt

import pymwm

params: dict = {
    "core": {"shape": "coax", "r": 0.15, "ri": 0.1, "fill": {"RI": 1.0}},
    "clad": {"book": "Au", "page": "Stewart-DLF", "bound_check": False},
    "bounds": {"wl_max": 5.0, "wl_min": 1.0, "wl_imag": 50.0},
    "modes": {"wl_max": 20.0, "wl_min": 1.0, "wl_imag": 50.0, "num_n": 6, "num_m": 2},
}


def test_attributes():
    wg = pymwm.create(params)
    assert wg.r == 0.15
    assert wg.ri == 0.1
    w = 2 * np.pi / 5.0
    assert wg.fill(w) == 1.0
    drude_lorentz = -1272.37592771801 + 351.25089220304176j
    npt.assert_almost_equal(wg.clad(w), drude_lorentz)
