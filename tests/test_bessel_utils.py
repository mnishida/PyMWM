from __future__ import annotations

import numpy as np
import numpy.testing as npt
import scipy.special as ssp

from pymwm.utils.bessel_utils import yv_yvp_cython


def test_yv_yvp():
    for n in range(4):
        z = np.random.random() + 1j * np.random.random()
        y1, yp1 = yv_yvp_cython(n, z)
        y2 = ssp.yv(n, z)
        yp2 = ssp.yvp(n, z)
        npt.assert_almost_equal(y1, y2)
        npt.assert_almost_equal(yp1, yp2)
