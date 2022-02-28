import numpy as np
import numpy.testing as npt

from pymwm.utils import eig_mat_utils


def det_func(a0, z):
    a = a0 + np.array([[z, 0, 0, z], [0, z, 0, 0], [0, 0, z, 0], [-1, 0, z, 0]])
    b = np.array(
        [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]], dtype=complex
    )
    return np.array(a), np.array(b)


def deriv_det_func(a0, z):
    return (
        np.linalg.det(det_func(a0, z + 1e-4)[0])
        - np.linalg.det(det_func(a0, z - 1e-4)[0])
    ) / 2e-4


def test_solve_cython():
    a = np.random.random((3, 3)) + np.random.random((3, 3)) * 1j
    b = np.random.random(3) + np.random.random(3) * 1j
    val1 = np.linalg.solve(a, b)
    val2 = eig_mat_utils.solve_cython(a, b)
    print(val1, val2)
    npt.assert_almost_equal(val1, val2)


def test_det4():
    a = np.cos(3 * (np.random.random((4, 4)) + np.random.random((4, 4)) * 1j))
    val1 = np.linalg.det(a)
    val2 = eig_mat_utils.det4_cython(a)
    print(val1, val2)
    npt.assert_almost_equal(val1, val2)


def test_deriv_det4():
    from scipy.sparse.linalg import eigs

    a0 = np.random.random((4, 4)) + np.random.random((4, 4)) * 1j
    z = np.random.random() + np.random.random() * 1j
    a, b = det_func(a0, z)
    a1, _ = det_func(a0, z + 1e-4)
    a2, _ = det_func(a0, z - 1e-4)
    da_dz = (a1 - a2) / 2e-4
    val1 = eig_mat_utils.deriv_det4_cython(a, b)
    val2 = deriv_det_func(a0, z)
    print(f"{a}")
    print(f"{b}")
    print(f"{da_dz}")
    print(f"{np.linalg.det(a)}")
    print(f"{eigs(a, k=1, which = 'SM', return_eigenvectors=False)}")
    print(f"{np.linalg.eigvals(a)}")
    s, val = np.linalg.slogdet(a)
    print(f"{s * np.exp(val)}")
    print(f"{eig_mat_utils.det4_cython(a)}")
    print(f"{val1}")
    print(f"{val2}")
    npt.assert_allclose(b, da_dz)
    npt.assert_almost_equal(val1, val2)


def test_deriv_det2():
    from scipy.linalg import det

    a0 = np.random.random((2, 2)) + np.random.random((2, 2)) * 1j
    z = np.random.random() + np.random.random() * 1j
    a = a0 + np.array([[z, 0], [0, z]])
    b = np.array([[1, 0], [0, 1]], dtype=complex)
    a1 = a0 + np.array([[z + 1e-4, 0], [0, z + 1e-4]])
    a2 = a0 + np.array([[z - 1e-4, 0], [0, z - 1e-4]])
    det_a1 = np.linalg.det(a1)
    det_a2 = np.linalg.det(a2)
    print(a1, a2)
    print(det_a1, det_a2)
    print(np.linalg.det(a2))
    print(det(a1))
    print(det(a2))
    val1 = eig_mat_utils.deriv_det2_cython(a, b)
    val2 = (det_a1 - det_a2) / 2e-4
    print(val1, val2)
    npt.assert_allclose(val1, val2)
