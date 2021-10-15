import numpy as np
import numpy.testing as npt

from pymwm.cylinder.samples import Samples
from pymwm.utils import cylinder_utils


def test_eiq_eq():
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
    roots = np.array([-1550 + 20j, -500 + 1.5j], dtype=complex)
    wg = Samples(size, fill, clad, p)
    w = 2 * np.pi / 100.0
    e1 = wg.fill(w)
    e2 = wg.clad(w) * 1000
    h2vec = np.array([-35513.94604091, -6.53379717e5])
    f1, fp1 = wg.eig_eq(h2vec, w, "E", 0, e1, e2, roots)
    f2, fp2 = cylinder_utils.eig_eq_cython(h2vec, w, "E", 0, e1, e2, size, roots)
    npt.assert_almost_equal(f1, f2)
    npt.assert_almost_equal(fp1, fp2)
    f1, fp1 = wg.eig_eq(h2vec, w, "M", 0, e1, e2, roots)
    f2, fp2 = cylinder_utils.eig_eq_cython(h2vec, w, "M", 0, e1, e2, size, roots)
    npt.assert_almost_equal(f1, f2)
    npt.assert_almost_equal(fp1, fp2)
    f1, fp1 = wg.eig_eq(h2vec, w, "M", 1, e1, e2, roots)
    f2, fp2 = cylinder_utils.eig_eq_cython(h2vec, w, "M", 1, e1, e2, size, roots)
    print(f1, fp1)
    npt.assert_almost_equal(f1, f2)
    npt.assert_almost_equal(fp1, fp2)


def test_eiq_eq_derivative():
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
    roots = np.array([-1550 + 20j, -500 + 1.5j], dtype=complex)
    wg = Samples(size, fill, clad, p)
    w = 2 * np.pi / 100.0
    e1 = wg.fill(w)
    e2 = wg.clad(w) * 1000
    h2vec = np.array([-35513.94604091, -6.53379717e5])
    eps = 1e-4
    f, fp = cylinder_utils.eig_eq_cython(h2vec, w, "E", 0, e1, e2, size, roots)
    f1, _ = cylinder_utils.eig_eq_cython(
        h2vec + np.array([eps, 0]), w, "E", 0, e1, e2, size, roots
    )
    f2, _ = cylinder_utils.eig_eq_cython(
        h2vec - np.array([eps, 0]), w, "E", 0, e1, e2, size, roots
    )
    f1 = f1[0] + f1[1] * 1j
    f2 = f2[0] + f2[1] * 1j
    fp1 = fp[0, 0] + fp[1, 0] * 1j
    fp2 = (f1 - f2) / (2 * eps)
    print(fp1, fp2)
    npt.assert_almost_equal(fp1, fp2)
