# -*- coding: utf-8 -*-
# cython: profile=False
import numpy as np

from pymwm.utils import eig_mat_utils

cimport cython
cimport numpy as np


@cython.cdivision(True)
cdef cdouble u_func(cdouble h2, cdouble w, cdouble e1, double r):
    return (1 + 1j) * csqrt(-0.5j * (e1 * w ** 2 - h2)) * r


@cython.cdivision(True)
cdef cdouble v_func(cdouble h2, cdouble w, cdouble e2, double r):
    return (1 - 1j) * csqrt(0.5j * (-e2 * w ** 2 + h2)) * r


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void eig_mat_with_deriv(
    cdouble h2, cdouble w, str pol, int n, cdouble e1, cdouble e2, double r, double ri, cdouble[:, ::1] a, cdouble[:, ::1] b
):
    cdef:
        cdouble vals[3]
        cdouble w2 = w ** 2
        cdouble hew = h2 / e2 / w2
        cdouble ee = e1 / e2
        cdouble x = u_func(h2, w, e1, ri)
        cdouble y = v_func(h2, w, e2, ri)
        cdouble u = u_func(h2, w, e1, r)
        cdouble v = v_func(h2, w, e2, r)
        cdouble ju, jpu, jppu, yu, ypu, yppu, jx, jpx, jppx, yx, ypx, yppx
        cdouble kv, kpv, kppv, iy, ipy, ippy
        cdouble da_du[4][4]
        cdouble da_dv[4][4]
        cdouble da_dx[4][4]
        cdouble da_dy[4][4]
        int sign

    if cimag(u) > 0:
        sign = 1
    elif cimag(u) == 0:
        sign = 0
    else:
        sign = -1

    jve_jvpe_jvppe(n, u, vals)
    ju = vals[0]
    jpu = vals[1]
    jppu = vals[2]
    yve_yvpe_yvppe(n, u, vals)
    yu = vals[0]
    ypu = vals[1]
    yppu = vals[2]
    kve_kvpe_kvppe(n, v, vals)
    kv = vals[0]
    kpv = vals[1]
    kppv = vals[2]
    jve_jvpe_jvppe(n, x, vals)
    jx = vals[0]
    jpx = vals[1]
    jppx = vals[2]
    yve_yvpe_yvppe(n, x, vals)
    yx = vals[0]
    ypx = vals[1]
    yppx = vals[2]
    ive_ivpe_ivppe(n, y, vals)
    iy = vals[0]
    ipy = vals[1]
    ippy = vals[2]

    cdef:
        int i, j
        cdouble du_dh2 = - r ** 2 / (2 * u)
        cdouble dv_dh2 = r ** 2 / (2 * v)
        cdouble dx_dh2 = - ri ** 2 / (2 * x)
        cdouble dy_dh2 = ri ** 2 / (2 * y)
        cdouble nuv = n * (v / u + u / v)
        cdouble dnuv_du = n * (- v / u ** 2 + 1 / v)
        cdouble dnuv_dv = n * (- u / v ** 2 + 1 / u)
        cdouble nxy = n * (y / x + x / y)
        cdouble dnxy_dx = n * (- y / x ** 2 + 1 / y)
        cdouble dnxy_dy = n * (- x / y ** 2 + 1 / x)

    a[0, 0] = jpu * kv * v + kpv * ju * u
    a[0, 1] = ypu * kv * v + kpv * yu * u
    a[0, 2] = nuv * ju * kv
    a[0, 3] = nuv * yu * kv
    a[1, 0] = jpx / yx * y + ipy / iy * jx / yx * x
    a[1, 1] = ypx / yx * y + ipy / iy * x
    a[1, 2] = nxy * jx / yx
    a[1, 3] = nxy
    a[2, 0] = hew * nuv * ju * kv
    a[2, 1] = hew * nuv * yu * kv
    a[2, 2] = ee * jpu * kv * v + kpv * ju * u
    a[2, 3] = ee * ypu * kv * v + kpv * yu * u
    a[3, 0] = hew * nxy * jx / yx
    a[3, 1] = hew * nxy
    a[3, 2] = ee * jpx / yx * y + ipy / iy * jx / yx * x
    a[3, 3] = ee * ypx / yx * y + ipy / iy * x


    da_du[0][0] = jppu * kv * v + kpv * (jpu * u + ju) + 1j * sign * a[0, 0]
    da_du[0][1] = yppu * kv * v + kpv * (ypu * u + yu) + 1j * sign * a[0, 1]
    da_du[0][2] = dnuv_du * ju * kv + nuv * jpu * kv + 1j * sign * a[0, 2]
    da_du[0][3] = dnuv_du * yu * kv + nuv * ypu * kv + 1j * sign * a[0, 3]
    da_du[2][0] = hew * (dnuv_du * ju + nuv * jpu) * kv + 1j * sign * a[2, 0]
    da_du[2][1] = hew * (dnuv_du * yu + nuv * ypu) * kv + 1j * sign * a[2, 1]
    da_du[2][2] = ee * jppu * kv * v + kpv * (jpu * u + ju) + 1j * sign * a[2, 2]
    da_du[2][3] = ee * yppu * kv * v + kpv * (ypu * u + yu) + 1j * sign * a[2, 3]
    for i in range(4):
        da_du[1][i] = da_du[3][i] = 0.0

    da_dv[0][0] = jpu * (kpv * v + kv) + kppv * ju * u + a[0, 0]
    da_dv[0][1] = ypu * (kpv * v + kv) + kppv * yu * u + a[0, 1]
    da_dv[0][2] = (dnuv_dv * kv + nuv * kpv) * ju + a[0, 2]
    da_dv[0][3] = (dnuv_dv * kv + nuv * kpv) * yu + a[0, 3]
    for i in range(4):
        da_dv[1][i] = da_dv[3][i] = 0.0
    da_dv[2][0] = hew * (dnuv_dv * kv + nuv * kpv) * ju + a[2, 0]
    da_dv[2][1] = hew * (dnuv_dv * kv + nuv * kpv) * yu + a[2, 1]
    da_dv[2][2] = ee * jpu * (kpv * v + kv) + kppv * ju * u + a[2, 2]
    da_dv[2][3] = ee * ypu * (kpv * v + kv) + kppv * yu * u + a[2, 3]

    for i in range(4):
        da_dx[0][i] = da_dx[2][i] = 0.0
    da_dx[1][0] = ((jppx / yx - jpx * ypx / yx ** 2) * y
        + ipy / iy * ((jpx / yx - jx * ypx / yx ** 2) * x + jx / yx))
    da_dx[1][1] = (yppx / yx - ypx ** 2 / yx ** 2) * y + ipy / iy
    da_dx[1][2] = dnxy_dx * jx / yx + nxy * jpx / yx - nxy * jx * ypx / yx ** 2
    da_dx[1][3] = dnxy_dx
    da_dx[3][0] = hew * (dnxy_dx * jx / yx + nxy * jpx / yx - nxy * jx * ypx / yx ** 2)
    da_dx[3][1] = hew * dnxy_dx
    da_dx[3][2] = (ee * (jppx / yx - jpx * ypx / yx ** 2) * y
        + ipy / iy * ((jpx / yx - jx * ypx / yx ** 2) * x + jx / yx))
    da_dx[3][3] = ee * (yppx / yx - ypx ** 2 / yx ** 2) * y + ipy / iy

    for i in range(4):
        da_dy[0][i] = da_dy[2][i] = 0.0
    da_dy[1][0] = jpx / yx + (ippy / iy - ipy ** 2 / iy ** 2) * jx / yx * x
    da_dy[1][1] = ypx / yx + (ippy / iy - ipy ** 2 / iy ** 2) * x
    da_dy[1][2] = dnxy_dy * jx / yx
    da_dy[1][3] = dnxy_dy
    da_dy[3][0] = hew * dnxy_dy * jx / yx
    da_dy[3][1] = hew * dnxy_dy
    da_dy[3][2] = ee * jpx / yx + (ippy / iy - ipy ** 2 / iy ** 2) * jx / yx * x
    da_dy[3][3] = ee * ypx / yx + (ippy / iy - ipy ** 2 / iy ** 2) * x

    for i in range(4):
        for j in range(4):
            b[i, j] = (da_du[i][j] * du_dh2 + da_dv[i][j] * dv_dh2
                        + da_dx[i][j] * dx_dh2 + da_dy[i][j] * dy_dh2)
    b[2, 0] += ee / w2 * nuv * ju * kv
    b[2, 1] += ee / w2 * nuv * yu * kv
    b[3, 0] += ee / w2 * nxy * jx / yx
    b[3, 1] += ee / w2 * nxy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def eig_eq_with_jac(
    double[::1] h2vec, cdouble w, str pol, int n, cdouble e1, cdouble e2, double r, double ri, cdouble[::1] roots
) -> tuple[np.ndarray, np.ndarray]:
    """Return the value of the characteristic equation

    Args:
        h2vec: The real and imaginary parts of the square of propagation constant.
        w: The angular frequency
        pol: The polarization
        n: The order of the modes
        e1: The permittivity of the core
        e2: The permittivity of the clad.
    Returns:
        val: A complex indicating the left-hand value of the characteristic
            equation.
    """
    cdef:
        cdouble h2 = h2vec[0] + h2vec[1] * 1j
        cdouble[:, ::1] a = np.empty((4, 4), dtype=complex)
        cdouble[:, ::1] b = np.empty((4, 4), dtype=complex)
        double norm
        cdouble f, fp, dd, ddi, denom
        int i
        int num = len(roots)
        cdouble[::1] tanhs = np.empty(num, dtype=complex)

    eig_mat_with_deriv(h2, w, pol, n, e1, e2, r, ri, a, b)
    if n == 0:
        if pol == "E":
            f = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
            fp = deriv_det2(
                np.ascontiguousarray(a[:2, :2]),
                np.ascontiguousarray(b[:2, :2]))
        else:
            f = a[2, 2] * a[3, 3] - a[2, 3] * a[3, 2]
            fp = deriv_det2(
                np.ascontiguousarray(a[2:, 2:]),
                np.ascontiguousarray(b[2:, 2:]))
    else:
        f = det4(a)
        fp = deriv_det4(a, b)
    denom = 1.0
    dd = 0.0
    for i in range(num):
        tanhs[i] = ctanh(h2 - roots[i])
    for i in range(num):
        denom *= tanhs[i]
        ddi = (tanhs[i] ** 2 - 1) / tanhs[i] ** 2
        for j in range(num):
            if j != i:
                ddi /= tanhs[j]
        dd += ddi
    fp = fp / denom + f * dd
    f /= denom
    return np.array([f.real, f.imag]), np.array([[fp.real, fp.imag], [-fp.imag, fp.real]])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def eig_eq_for_min(
    double[::1] h2vec, cdouble w, str pol, int n, cdouble e1, cdouble e2, double r, double ri, cdouble[::1] roots
) -> tuple[float, np.ndarray]:
    """Return the value of the characteristic equation

    Args:
        h2vec: The real and imaginary parts of the square of propagation constant.
        w: The angular frequency
        pol: The polarization
        n: The order of the modes
        e1: The permittivity of the core
        e2: The permittivity of the clad.
    Returns:
        val: A complex indicating the left-hand value of the characteristic
            equation.
    """
    cdef:
        cdouble h2 = h2vec[0] + h2vec[1] * 1j
        cdouble[:, ::1] a = np.empty((4, 4), dtype=complex)
        cdouble[:, ::1] b = np.empty((4, 4), dtype=complex)
        double norm
        cdouble f, fp, dd, ddi, denom
        int i, j
        int num = len(roots)
        cdouble[::1] tanhs = np.empty(num, dtype=complex)

    eig_mat_with_deriv(h2, w, pol, n, e1, e2, r, ri, a, b)
    if n == 0:
        if pol == "E":
            f = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
            fp = deriv_det2(
                np.ascontiguousarray(a[:2, :2]),
                np.ascontiguousarray(b[:2, :2]))
        else:
            f = a[2, 2] * a[3, 3] - a[2, 3] * a[3, 2]
            fp = deriv_det2(
                np.ascontiguousarray(a[2:, 2:]),
                np.ascontiguousarray(b[2:, 2:]))
    else:
        f = det4(a)
        fp = deriv_det4(a, b)
    denom = 1.0
    dd = 0.0
    for i in range(num):
        tanhs[i] = ctanh(h2 - roots[i])
    for i in range(num):
        denom *= tanhs[i]
        ddi = (tanhs[i] ** 2 - 1) / tanhs[i] ** 2
        for j in range(num):
            if j != i:
                ddi /= tanhs[j]
        dd += ddi
    fp = fp / denom + f * dd
    f /= denom
    return (f.real ** 2 + f.imag ** 2,
            np.array([
                2 * (f.real * fp.real + f.imag * fp.imag),
                2 * (-f.real * fp.imag + f.imag * fp.real)]))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void eig_mat(
    cdouble h2, cdouble w, str pol, int n, cdouble e1, cdouble e2, double r, double ri, cdouble[:, ::1] a
):
    cdef:
        cdouble vals[3]
        cdouble w2 = w ** 2
        cdouble hew = h2 / e2 / w2
        cdouble ee = e1 / e2
        cdouble x = u_func(h2, w, e1, ri)
        cdouble y = v_func(h2, w, e2, ri)
        cdouble u = u_func(h2, w, e1, r)
        cdouble v = v_func(h2, w, e2, r)
        cdouble ju, jpu, jppu, yu, ypu, yppu, jx, jpx, jppx, yx, ypx, yppx
        cdouble kv, kpv, kppv, iy, ipy, ippy
        cdouble nuv = n * (v / u + u / v)
        cdouble nxy = n * (y / x + x / y)

    jve_jvpe_jvppe(n, u, vals)
    ju = vals[0]
    jpu = vals[1]
    jppu = vals[2]
    yve_yvpe_yvppe(n, u, vals)
    yu = vals[0]
    ypu = vals[1]
    yppu = vals[2]
    kve_kvpe_kvppe(n, v, vals)
    kv = vals[0]
    kpv = vals[1]
    kppv = vals[2]
    jve_jvpe_jvppe(n, x, vals)
    jx = vals[0]
    jpx = vals[1]
    jppx = vals[2]
    yve_yvpe_yvppe(n, x, vals)
    yx = vals[0]
    ypx = vals[1]
    yppx = vals[2]
    ive_ivpe_ivppe(n, y, vals)
    iy = vals[0]
    ipy = vals[1]
    ippy = vals[2]

    a[0, 0] = jpu * kv * v + kpv * ju * u
    a[0, 1] = ypu * kv * v + kpv * yu * u
    a[0, 2] = nuv * ju * kv
    a[0, 3] = nuv * yu * kv
    a[1, 0] = jpx / yx * y + ipy / iy * jx / yx * x
    a[1, 1] = ypx / yx * y + ipy / iy * x
    a[1, 2] = nxy * jx / yx
    a[1, 3] = nxy
    a[2, 0] = hew * nuv * ju * kv
    a[2, 1] = hew * nuv * yu * kv
    a[2, 2] = ee * jpu * kv * v + kpv * ju * u
    a[2, 3] = ee * ypu * kv * v + kpv * yu * u
    a[3, 0] = hew * nxy * jx / yx
    a[3, 1] = hew * nxy
    a[3, 2] = ee * jpx / yx * y + ipy / iy * jx / yx * x
    a[3, 3] = ee * ypx / yx * y + ipy / iy * x


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def eig_eq(
    double[::1] h2vec, cdouble w, str pol, int n, cdouble e1, cdouble e2, double r, double ri, cdouble[::1] roots
) -> np.ndarray:
    """Return the value of the characteristic equation

    Args:
        h2vec: The real and imaginary parts of the square of propagation constant.
        w: The angular frequency
        pol: The polarization
        n: The order of the modes
        e1: The permittivity of the core
        e2: The permittivity of the clad.
    Returns:
        np.array([f.real, f.imag]):  Real and imaginary parts of left-hand value of the characteristic equation.
    """
    cdef:
        cdouble h2 = h2vec[0] + h2vec[1] * 1j
        cdouble[:, ::1] a = np.empty((4, 4), dtype=complex)
        double norm
        cdouble f, fp, denom
        int i
        int num = len(roots)

    eig_mat(h2, w, pol, n, e1, e2, r, ri, a)
    if n == 0:
        if pol == "E":
            f = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
        else:
            f = a[2, 2] * a[3, 3] - a[2, 3] * a[3, 2]
    else:
        f = det4(a)
    denom = 1.0
    for i in range(num):
        denom *= ctanh(h2 - roots[i])
    f /= denom
    return np.array([f.real, f.imag])
