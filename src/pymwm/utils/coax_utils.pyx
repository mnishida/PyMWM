# -*- coding: utf-8 -*-
# cython: profile=False
import numpy as np

cimport cython
cimport numpy as np


@cython.cdivision(True)
cdef cdouble u_func(cdouble h2, cdouble w, cdouble e1, double r) nogil:
    return (1 + 1j) * csqrt(-0.5j * (e1 * w ** 2 - h2)) * r


@cython.cdivision(True)
cdef cdouble v_func(cdouble h2, cdouble w, cdouble e2, double r) nogil:
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
        denom *= (h2 - roots[i]) / roots[i]
        ddi = - roots[i] / (h2 - roots[i]) ** 2
        for j in range(num):
            if j != i:
                ddi /= (h2 - roots[j]) / roots[j]
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
        denom *= (h2 - roots[i]) / roots[i]
        ddi = - roots[i] / (h2 - roots[i]) ** 2
        for j in range(num):
            if j != i:
                ddi /= (h2 - roots[j]) / roots[j]
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
        denom *= (h2 - roots[i]) / roots[i]
    f /= denom
    return np.array([f.real, f.imag])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void coefs_C(
    cdouble *hs, cdouble w, long *s_all, long *n_all,
    int num_n_all, double r, double ri, cdouble e1, cdouble e2,
    cdouble *xs, cdouble *ys, cdouble *us, cdouble *vs,
    cdouble *jxs, cdouble *jpxs, cdouble *yxs, cdouble *ypxs, cdouble *iys, cdouble *ipys,
    cdouble *jus, cdouble *jpus, cdouble *yus, cdouble *ypus, cdouble *kvs, cdouble *kpvs,
    cdouble *A1s, cdouble *B1s, cdouble *A2s, cdouble *B2s,
    cdouble *C2s, cdouble *D2s, cdouble *A3s, cdouble *B3s,
    cdouble *Ys) nogil:
    cdef:
        int i, s, n, en
        cdouble ee = e1 / e2
        cdouble val
        cdouble I1, I2, I3, Y1, Y2, Y3
        cdouble norm, Y, hew
        cdouble h, a1, b1, a2, b2, c2, d2, a3, b3
        cdouble x, y, u, v
        cdouble ju, jpu, yu, ypu, jx, jpx, yx, ypx
        cdouble kv, kpv, iy, ipy, nuv, nxy
        cdouble y_te, y_tm1, y_tm2
        cdouble A[3][3]
        cdouble B[3]
        cdouble C[3]
        cdouble vals[2]
    for i in range(num_n_all):
        n = n_all[i]
        en = 1 if n == 0 else 2
        h = hs[i]
        y_te = h / w
        y_tm1 = e1 * w / h
        y_tm2 = e2 * w / h
        s = s_all[i]
        u = u_func(h * h, w, e1, r)
        jv_jvp(n, u, vals)
        ju = vals[0]
        jpu = vals[1]
        yv_yvp(n, u, vals)
        yu = vals[0]
        ypu = vals[1]
        x = u_func(h * h, w, e1, ri)
        jv_jvp(n, x, vals)
        jx = vals[0]
        jpx = vals[1]
        yv_yvp(n, x, vals)
        yx = vals[0]
        ypx = vals[1]

        if creal(e2) < -1e6:
            a1 = b1 = a3 = b3 = 0.0j
            if s == 0:
                a2 = 1.0 + 0.0j
                c2 = - jpu / ypu
                b2 = d2 = 0.0j
            else:
                b2 = 1.0 + 0.0j
                d2 = - ju / yu
                a2 = c2 = 0.0j
            y = INFINITY
            v = INFINITY
            iys[i] = INFINITY
            ipys[i] = INFINITY
            kvs[i] = 0.0
            kpvs[i] = 0.0
        else:
            hew = h ** 2 / e2 / w ** 2
            y = v_func(h * h, w, e2, ri)
            v = v_func(h * h, w, e2, r)
            kv_kvp(n, v, vals)
            kv = vals[0]
            kpv = vals[1]
            iv_ivp(n, y, vals)
            iy = vals[0]
            ipy = vals[1]
            nuv = n * (v / u + u / v)
            nxy = n * (y / x + x / y)
            if s == 0:
                A[0][0] = ypx / yx * y + ipy / iy * x
                A[0][1] = nxy * jx / yx
                A[0][2] = nxy
                A[1][0] = hew * nuv * yu * kv
                A[1][1] = ee * jpu * kv * v + kpv * ju * u
                A[1][2] = ee * ypu * kv * v + kpv * yu * u
                A[2][0] = hew * nxy
                A[2][1] = ee * jpx / yx * y + ipy / iy * jx / yx * x
                A[2][2] = ee * ypx / yx * y + ipy / iy * x
                B[0] = - (jpx / yx * y + ipy / iy * jx / yx * x)
                B[1] = - (hew * nuv * ju * kv)
                B[2] = - (hew * nxy * jx / yx)
                a2 = 1.0 + 0j
                solve(A, B, C)
                c2 = C[0]
                b2 = C[1]
                d2 = C[2]
            else:
                A[0][0] = ee * ypx / yx * y + ipy / iy * x
                A[0][1] = hew * nxy * jx / yx
                A[0][2] = hew * nxy
                A[1][0] = nuv * yu * kv
                A[1][1] = jpu * kv * v + kpv * ju * u
                A[1][2] = ypu * kv * v + kpv * yu * u
                A[2][0] = nxy
                A[2][1] = jpx / yx * y + ipy / iy * jx / yx * x
                A[2][2] = ypx / yx * y + ipy / iy * x
                B[0] = - (ee * jpx / yx * y + ipy / iy * jx / yx * x)
                B[1] = - (nuv * ju * kv)
                B[2] = - (nxy * jx / yx)
                b2 = 1.0 + 0j
                solve(A, B, C)
                d2 = C[0]
                a2 = C[1]
                c2 = C[2]
            a1 = - x / (y * iy) * (jx * a2 + yx * c2)
            b1 = - x / (y * iy) * (jx * b2 + yx * d2)
            a3 = - u / (v * kv) * (ju * a2 + yu * c2)
            b3 = - u / (v * kv) * (ju * b2 + yu * d2)

        val = M_PI / en * (
            r ** 2 * (jpu ** 2 + (1 - n ** 2 / u ** 2) * ju ** 2 + 2 * jpu * ju / u)
            - ri ** 2 * (jpx ** 2 + (1 - n ** 2 / x ** 2) * jx ** 2 + 2 * jpx * jx / x)
        )
        I2 = val * (a2 ** 2 + b2 ** 2)
        Y2 = val * (y_te * a2 ** 2 + y_tm1 * b2 ** 2)
        val = M_PI / en * (
            r ** 2 * (ypu ** 2 + (1 - n ** 2 / u ** 2) * yu ** 2 + 2 * ypu * yu / u)
            - ri ** 2 * (ypx ** 2 + (1 - n ** 2 / x ** 2) * yx ** 2 + 2 * ypx * yx / x)
        )
        I2 += val * (c2 ** 2 + d2 ** 2)
        Y2 += val * (y_te * c2 ** 2 + y_tm1 * d2 ** 2)
        val = M_PI / en * 2 * (
            r ** 2 * (jpu * ypu + (1 - n ** 2 / u ** 2) * ju * yu + 2 * jpu * yu / u)
            - ri ** 2 * (jpx * ypx + (1 - n ** 2 / x ** 2) * jx * yx + 2 * jpx * yx / x)
        )
        I2 += val * (a2 * c2 + b2 * d2)
        Y2 += val * (y_te * a2 * c2 + y_tm1 * b2 * d2)
        val = M_PI * n * (r ** 2 / u ** 2 * ju ** 2 - ri ** 2 / x ** 2 * jx ** 2)
        I2 += val * 2 * a2 * b2
        Y2 += val * (y_te + y_tm1) * a2 * b2
        val = M_PI * n * (r ** 2 / u ** 2 * yu ** 2 - ri ** 2 / x ** 2 * yx ** 2)
        I2 += 2 * val * c2 * d2
        Y2 += val * (y_te + y_tm1) * c2 * d2
        val = M_PI * n *  (r ** 2 / u ** 2 * ju * yu - ri ** 2 / x ** 2 * jx * yx)
        I2 += 2 * val * (a2 * d2 + b2 * c2)
        Y2 +=  val * (y_te + y_tm1) * (a2 * d2 + b2 * c2)

        if creal(e2) < -1e6:
            norm = csqrt(I2)
            Y = Y2 / I2
        else:
            val = M_PI * ri ** 2 / en * (ipy ** 2 - (1 + n ** 2 / y ** 2) * iy ** 2 + 2 * ipy * iy / y)
            I1 = val * (a1 ** 2 + b1 ** 2)
            Y1 = val * (y_te * a1 ** 2 + y_tm2 * b1 ** 2)
            val = M_PI * ri ** 2 * n * iy ** 2 / y ** 2
            I1 += val * 2 * a1 * b1
            Y1 += val * (y_te + y_tm2) * a1 * b1
            val = - M_PI * r ** 2 / en * (kpv ** 2 - (1 + n ** 2 / v ** 2) * kv ** 2 + 2 * kpv * kv / v)
            I3 = val * (a3 ** 2 + b3 ** 2)
            Y3 = val * (y_te * a3 ** 2 + y_tm2 * b3 ** 2)
            val = - M_PI * r ** 2 * n * kv ** 2 / v ** 2
            I3 += val * 2 * a3 * b3
            Y3 += val * (y_te + y_tm2) * a3 * b3
            norm = csqrt(I1 + I2 + I3)
            Y = (Y1 + Y2 + Y3) / (I1 + I2 + I3)

        xs[i] = x
        ys[i] = y
        us[i] = u
        vs[i] = v
        jxs[i] = jx
        jpxs[i] = jpx
        yxs[i] = yx
        ypxs[i] = ypx
        iys [i] = iy
        ipys[i] = ipy
        jus[i] = ju
        jpus[i] = jpu
        yus[i] = yu
        ypus[i] = ypu
        kvs [i] = kv
        kpvs[i] = kpv
        A1s[i] = a1 / norm
        B1s[i] = b1 / norm
        A2s[i] = a2 / norm
        B2s[i] = b2 / norm
        C2s[i] = c2 / norm
        D2s[i] = d2 / norm
        A3s[i] = a3 / norm
        B3s[i] = b3 / norm
        Ys[i] = Y


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def props_cython(cdouble w, double r, double ri, long[::1] s_all, long[::1] n_all,
                 cdouble[::1] hs, cdouble e1, cdouble e2):
    cdef:
        int num_n_all = n_all.shape[0]
    xs_array = np.empty(num_n_all, dtype=complex)
    ys_array = np.empty(num_n_all, dtype=complex)
    us_array = np.empty(num_n_all, dtype=complex)
    vs_array = np.empty(num_n_all, dtype=complex)
    jxs_array = np.empty(num_n_all, dtype=complex)
    jpxs_array = np.empty(num_n_all, dtype=complex)
    yxs_array = np.empty(num_n_all, dtype=complex)
    ypxs_array = np.empty(num_n_all, dtype=complex)
    iys_array = np.empty(num_n_all, dtype=complex)
    ipys_array = np.empty(num_n_all, dtype=complex)
    jus_array = np.empty(num_n_all, dtype=complex)
    jpus_array = np.empty(num_n_all, dtype=complex)
    yus_array = np.empty(num_n_all, dtype=complex)
    ypus_array = np.empty(num_n_all, dtype=complex)
    kvs_array = np.empty(num_n_all, dtype=complex)
    kpvs_array = np.empty(num_n_all, dtype=complex)
    A1s_array = np.empty(num_n_all, dtype=complex)
    B1s_array = np.empty(num_n_all, dtype=complex)
    A2s_array = np.empty(num_n_all, dtype=complex)
    B2s_array = np.empty(num_n_all, dtype=complex)
    C2s_array = np.empty(num_n_all, dtype=complex)
    D2s_array = np.empty(num_n_all, dtype=complex)
    A3s_array = np.empty(num_n_all, dtype=complex)
    B3s_array = np.empty(num_n_all, dtype=complex)
    Ys_array = np.empty(num_n_all, dtype=complex)
    cdef:
        cdouble[:] xs = xs_array
        cdouble[:] ys = ys_array
        cdouble[:] us = us_array
        cdouble[:] vs = vs_array
        cdouble[:] jxs = jxs_array
        cdouble[:] jpxs = jpxs_array
        cdouble[:] yxs = yxs_array
        cdouble[:] ypxs = ypxs_array
        cdouble[:] iys = iys_array
        cdouble[:] ipys = ipys_array
        cdouble[:] jus = jus_array
        cdouble[:] jpus = jpus_array
        cdouble[:] yus = yus_array
        cdouble[:] ypus = ypus_array
        cdouble[:] kvs = kvs_array
        cdouble[:] kpvs = kpvs_array
        cdouble[:] A1s = A1s_array
        cdouble[:] B1s = B1s_array
        cdouble[:] A2s = A2s_array
        cdouble[:] B2s = B2s_array
        cdouble[:] C2s = C2s_array
        cdouble[:] D2s = D2s_array
        cdouble[:] A3s = A3s_array
        cdouble[:] B3s = B3s_array
        cdouble[:] Ys = Ys_array
    coefs_C(&hs[0], w, &s_all[0], &n_all[0], num_n_all, r, ri, e1, e2,
        &xs[0], &ys[0], &us[0], &vs[0], &jxs[0], &jpxs[0], &yxs[0], &ypxs[0],
        &iys[0], &ipys[0], &jus[0], &jpus[0], &yus[0], &ypus[0], &kvs[0], &kpvs[0],
        &A1s[0], &B1s[0], &A2s[0], &B2s[0], &C2s[0], &D2s[0], &A3s[0], &B3s[0], &Ys[0])
    return (xs_array, ys_array, us_array, vs_array,
            jxs_array, jpxs_array, yxs_array, ypxs_array, iys_array, ipys_array,
            jus_array, jpus_array, yus_array, ypus_array, kvs_array, kpvs_array,
            A1s_array, B1s_array, A2s_array, B2s_array, C2s_array, D2s_array, A3s_array, B3s_array, Ys_array)
