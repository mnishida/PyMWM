import numpy as np
import numpy.testing as npt

from pymwm.coax.samples import Samples
from pymwm.utils import coax_utils


def test_eig_mat():
    size = 0.15
    fill = {"RI": 1.0}
    clad = {"book": "Au", "page": "Stewart-DLF", "bound_check": False}
    p = {
        "wl_max": 10.0,
        "wl_min": 1.0,
        "wl_imag": 50.0,
        "dw": 1.0 / 64,
        "num_n": 6,
        "num_m": 2,
    }
    size2 = 0.1
    wg = Samples(size, fill, clad, p, size2)
    h2 = -3500 + 6.3379717e5j
    w = 2 * np.pi / 100.0
    e1 = wg.fill(w)
    e2 = wg.clad(w) * 1000
    eps = 1e-4
    # a = np.empty((4, 4), dtype=complex)
    # b = np.empty((4, 4), dtype=complex)
    # a1 = np.empty((4, 4), dtype=complex)
    # a2 = np.empty((4, 4), dtype=complex)
    # b1 = np.empty((4, 4), dtype=complex)
    # coax_utils.eig_mat_with_deriv(h2, w, "M", 0, e1, e2, size, size2, a, b)
    # coax_utils.eig_mat_with_deriv(h2 + eps, w, "M", 0, e1, e2, size, size2, a1, b1)
    # coax_utils.eig_mat_with_deriv(h2 - eps, w, "M", 0, e1, e2, size, size2, a2, b1)
    a, b = wg.eig_mat(h2, w, "M", 0, e1, e2)
    a1, _ = wg.eig_mat(h2 + eps, w, "M", 0, e1, e2)
    a2, _ = wg.eig_mat(h2 - eps, w, "M", 0, e1, e2)
    da_dh2_1 = (a1 - a2) / (2 * eps)
    a3, _ = wg.eig_mat(h2 + eps * 1j, w, "M", 0, e1, e2)
    a4, _ = wg.eig_mat(h2 - eps * 1j, w, "M", 0, e1, e2)
    da_dh2_2 = -1j * (a3 - a4) / (2 * eps)
    # print(f"{a}")
    # print(f"{b}")
    # print(f"{da_dh2_1}")
    # print(f"{da_dh2_2}")
    npt.assert_almost_equal(b, da_dh2_1)
    npt.assert_almost_equal(b, da_dh2_2)
    a, b = wg.eig_mat(h2, w, "E", 0, e1, e2)
    a1, _ = wg.eig_mat(h2 + eps, w, "E", 0, e1, e2)
    a2, _ = wg.eig_mat(h2 - eps, w, "E", 0, e1, e2)
    da_dh2_1 = (a1 - a2) / (2 * eps)
    a3, _ = wg.eig_mat(h2 + eps * 1j, w, "E", 0, e1, e2)
    a4, _ = wg.eig_mat(h2 - eps * 1j, w, "E", 0, e1, e2)
    da_dh2_2 = -1j * (a3 - a4) / (2 * eps)
    # print(f"{a}")
    # print(f"{b}")
    # print(f"{da_dh2_1}")
    # print(f"{da_dh2_2}")
    npt.assert_almost_equal(b, da_dh2_1)
    npt.assert_almost_equal(b, da_dh2_2)
    # coax_utils.eig_mat_with_deriv(h2, w, "M", 1, e1, e2, size, size2, a, b)
    # coax_utils.eig_mat_with_deriv(h2 + eps, w, "M", 1, e1, e2, size, size2, a1, b1)
    # coax_utils.eig_mat_with_deriv(h2 - eps, w, "M", 1, e1, e2, size, size2, a2, b1)
    a, b = wg.eig_mat(h2, w, "M", 1, e1, e2)
    a1, _ = wg.eig_mat(h2 + eps, w, "M", 1, e1, e2)
    a2, _ = wg.eig_mat(h2 - eps, w, "M", 1, e1, e2)
    da_dh2_1 = (a1 - a2) / (2 * eps)
    a3, _ = wg.eig_mat(h2 + eps * 1j, w, "M", 1, e1, e2)
    a4, _ = wg.eig_mat(h2 - eps * 1j, w, "M", 1, e1, e2)
    da_dh2_2 = -1j * (a3 - a4) / (2 * eps)
    # print(f"{a}")
    print(f"{b}")
    print(f"{da_dh2_1}")
    print(f"{da_dh2_2}")
    npt.assert_almost_equal(b, da_dh2_1)
    npt.assert_almost_equal(b, da_dh2_2)

    h2 = -3500 - 6.3379717e5j
    a, b = wg.eig_mat(h2, w, "M", 0, e1, e2)
    a1, _ = wg.eig_mat(h2 + eps, w, "M", 0, e1, e2)
    a2, _ = wg.eig_mat(h2 - eps, w, "M", 0, e1, e2)
    da_dh2_1 = (a1 - a2) / (2 * eps)
    a3, _ = wg.eig_mat(h2 + eps * 1j, w, "M", 0, e1, e2)
    a4, _ = wg.eig_mat(h2 - eps * 1j, w, "M", 0, e1, e2)
    da_dh2_2 = -1j * (a3 - a4) / (2 * eps)
    # print(f"{a}")
    # print(f"{b}")
    # print(f"{da_dh2_1}")
    # print(f"{da_dh2_2}")
    npt.assert_almost_equal(b, da_dh2_1)
    npt.assert_almost_equal(b, da_dh2_2)
    a, b = wg.eig_mat(h2, w, "E", 0, e1, e2)
    a1, _ = wg.eig_mat(h2 + eps, w, "E", 0, e1, e2)
    a2, _ = wg.eig_mat(h2 - eps, w, "E", 0, e1, e2)
    da_dh2_1 = (a1 - a2) / (2 * eps)
    a3, _ = wg.eig_mat(h2 + eps * 1j, w, "E", 0, e1, e2)
    a4, _ = wg.eig_mat(h2 - eps * 1j, w, "E", 0, e1, e2)
    da_dh2_2 = -1j * (a3 - a4) / (2 * eps)
    # print(f"{a}")
    # print(f"{b}")
    # print(f"{da_dh2_1}")
    # print(f"{da_dh2_2}")
    npt.assert_almost_equal(b, da_dh2_1)
    npt.assert_almost_equal(b, da_dh2_2)
    a, b = wg.eig_mat(h2, w, "M", 1, e1, e2)
    a1, _ = wg.eig_mat(h2 + eps, w, "M", 1, e1, e2)
    a2, _ = wg.eig_mat(h2 - eps, w, "M", 1, e1, e2)
    da_dh2_1 = (a1 - a2) / (2 * eps)
    a3, _ = wg.eig_mat(h2 + eps * 1j, w, "M", 1, e1, e2)
    a4, _ = wg.eig_mat(h2 - eps * 1j, w, "M", 1, e1, e2)
    da_dh2_2 = -1j * (a3 - a4) / (2 * eps)
    # print(f"{a}")
    print(f"{b}")
    print(f"{da_dh2_1}")
    print(f"{da_dh2_2}")
    npt.assert_almost_equal(b, da_dh2_1)
    npt.assert_almost_equal(b, da_dh2_2)
