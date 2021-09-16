# -*- coding: utf-8 -*-
# cython: profile=False
import numpy as np

cimport cython
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void jk_to_coefs(int n, int pol, cdouble h,
                      cdouble u, cdouble jnu, cdouble jnpu,
                      cdouble v, cdouble knv, cdouble knpv,
                      cdouble w, double r,
                      cdouble e1, cdouble e2,
                      cdouble ab[2]) nogil:
    cdef:
        double norm
        cdouble c
    c = - n * (u * u + v * v) * jnu * knv / (u * v)
    if pol == 0:
        c *= h * h / (w * w)
        c /= (e1 * jnpu * v * knv + e2 * knpv * u * jnu)
        # norm = sqrt(1.0 + cabs(c) * cabs(c))
        # ab[0] = 1.0 / norm
        # ab[1] = c / norm
        ab[0] = 1.0
        ab[1] = c
    else:
        c /= (jnpu * v * knv + knpv * u * jnu)
        # norm = sqrt(1.0 + cabs(c) * cabs(c))
        # ab[1] = 1.0 / norm
        # ab[0] = c / norm
        ab[1] = 1.0
        ab[0] = c


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef cdouble upart_diag(int n, cdouble uc, cdouble jnuc, cdouble jnpuc,
                        cdouble u, cdouble jnu, cdouble jnpu) nogil:
    cdef:
        int n2 = n * n
        int sign
        cdouble u2, jnu2, jnpu2, u0, jnu0, jnpu0
        cdouble vals[2]
    if cabs(uc - u) < 1e-10:
        u0 = (u + uc) / 2
        sign = 1
        jv_jvp(n, u0, vals)
        jnu0 = vals[0]
        jnpu0 = vals[1]
        u2 = u0 * u0
        jnu2 = jnu0 * jnu0
        jnpu2 = jnpu0 * jnpu0
        return (sign *
                (jnu0 * jnpu0 / u0 + (jnpu2 + (1.0 - n2 / u2) * jnu2) / 2.0))
    elif cabs(uc + u) < 1e-10:
        u0 = (u - uc) / 2
        sign = 1 - ((n - 1) % 2) * 2
        jv_jvp(n, u0, vals)
        jnu0 = vals[0]
        jnpu0 = vals[1]
        u2 = u0 * u0
        jnu2 = jnu0 * jnu0
        jnpu2 = jnpu0 * jnpu0
        return (sign *
                (jnu0 * jnpu0 / u0 + (jnpu2 + (1.0 - n2 / u2) * jnu2) / 2.0))
    else:
        u2 = u * u
        jnu2 = jnu * jnu
        jnpu2 = jnpu * jnpu
        return (uc * jnuc * jnpu - u * jnu * jnpuc) / (uc * uc - u2)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline cdouble upart_off(int n, cdouble uc, cdouble jnuc,
                              cdouble u, cdouble jnu) nogil:
    return n * (jnuc * jnu) / (uc * u)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef cdouble vpart_diag(int n, cdouble vc, cdouble knvc, cdouble knpvc,
                        cdouble v, cdouble knv, cdouble knpv) nogil:
    cdef:
        int n2 = n * n
        int sign
        cdouble v2, knv2, knpv2, v0, knv0, knpv0
        cdouble vals[2]
    if cabs(vc - v) < 1e-10:
        v0 = (v + vc) / 2
        sign = 1
        kv_kvp(n, v0, vals)
        knv0 = vals[0]
        knpv0 = vals[1]
        v2 = v0 * v0
        knv2 = knv0 * knv0
        knpv2 = knpv0 * knpv0
        return sign * (
            knv0 * knpv0 / v0 +
            (knpv2 - (1.0 + n2 / v2) * knv2) / 2.0)
    elif cabs(vc + v) < 1e-10:
        v0 = (v - vc) / 2
        sign = 1 - ((n - 1) % 2) * 2
        kv_kvp(n, v0, vals)
        knv0 = vals[0]
        knpv0 = vals[1]
        v2 = v0 * v0
        knv2 = knv0 * knv0
        knpv2 = knpv0 * knpv0
        return sign * (
            knv0 * knpv0 / v0 +
            (knpv2 - (1.0 + n2 / v2) * knv2) / 2.0)
    else:
        v2 = v * v
        knv2 = knv * knv
        knpv2 = knpv * knpv
        return (vc * knvc * knpv -
                v * knv * knpvc) / (vc * vc - v2)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline cdouble vpart_off(int n, cdouble vc, cdouble knvc,
                              cdouble v, cdouble knv) nogil:
    return n * (knvc * knv) / (vc * v)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def coefs_cython(object hole, cdouble[::1] hs, cdouble w):
    cdef:
        long[::1] s_all = hole.s_all
        long[::1] n_all = hole.n_all
        long[::1] m_all = hole.m_all
    As_array = np.empty(hole.num_n_all, dtype=complex)
    Bs_array = np.empty(hole.num_n_all, dtype=complex)
    cdef:
        cdouble[::1] As = As_array
        cdouble[::1] Bs = Bs_array
        cdouble e1 = hole.fill(w)
        cdouble e2 = hole.clad(w)
    if creal(e2) < -1e6:
        coefs_pec_C(
            &s_all[0], &n_all[0], &m_all[0], hole.num_n_all,
            hole.r, &As[0], &Bs[0])
    else:
        coefs_C(
            &hs[0], w, &s_all[0], &n_all[0], &m_all[0], hole.num_n_all,
            hole.r, e1, e2, &As[0], &Bs[0])
    return As_array, Bs_array


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void coefs_pec_C(long *s_all, long *n_all, long  *m_all, int num_n_all,
              double r, cdouble *As, cdouble *Bs):
    cdef:
        int i, n, m, en
        double u, norm, jnu, jnpu
    from scipy.special import jn_zeros, jnp_zeros, jv, jvp
    for i in range(num_n_all):
        n = n_all[i]
        m = m_all[i]
        en = 1 if n == 0 else 2
        if s_all[i] == 0:
            u = jnp_zeros(n, m)[m - 1]
            jnu = jv(n, u)
            norm = sqrt(M_PI * r * r / en  *
                         (1 - n * n / (u * u)) * jnu * jnu)
            As[i] = 1.0 / norm
            Bs[i] = 0.0
        else:
            u = jn_zeros(n, m)[m - 1]
            jnpu = jvp(n, u)
            norm = sqrt(M_PI * r * r / en * jnpu * jnpu)
            As[i] = 0.0
            Bs[i] = 1.0 / norm


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void coefs_C(
    cdouble *hs, cdouble w, long *s_all, long *n_all, long  *m_all,
    int num_n_all, double r, cdouble e1, cdouble e2,
    cdouble *As, cdouble *Bs) nogil:
    cdef:
        int i, s, n, m, en
        cdouble norm
        cdouble h, u, v, jnu, jnpu, knv, knpv, a, b
        cdouble ab[2]
        cdouble vals[2]
        cdouble val_u, val_v, ud, uod, vd, vod
    for i in range(num_n_all):
        n = n_all[i]
        en = 1 if n == 0 else 2
        h = hs[i]
        s = s_all[i]
        u = (1 + 1j) * csqrt(-0.5j * (e1 * w * w - h * h)) * r
        jv_jvp(n, u, vals)
        jnu = vals[0]
        jnpu = vals[1]
        v = (1 - 1j) * csqrt(0.5j * (- e2 * w * w + h * h)) * r
        kv_kvp(n, v, vals)
        knv = vals[0]
        knpv = vals[1]
        jk_to_coefs(n, s, h, u, jnu, jnpu, v, knv, knpv, w, r, e1, e2, ab)
        a = ab[0]
        b = ab[1]
        val_u = 2 * M_PI * r * r / en
        val_v = (u * jnu) / (v * knv)
        val_v *= val_u * val_v
        ud = upart_diag(n, u, jnu, jnpu, u, jnu, jnpu)
        vd = vpart_diag(n, v, knv, knpv, v, knv, knpv)
        uod = upart_off(n, u, jnu, u, jnu)
        vod = vpart_off(n, v, knv, v, knv)
        norm = csqrt(val_u * (
            a * (a * ud + b * uod) + b * (b * ud + a * uod)) - val_v * (
            a * (a * vd + b * vod) + b * (b * vd + a * vod)))
        As[i] = a / norm
        Bs[i] = b / norm


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def ABY_cython(cdouble w, double r, long[::1] s_all, long[::1] n_all,
               long[::1] m_all, cdouble[::1] hs, cdouble e1, cdouble e2,
               double[:, :, ::1] u_pec, double[:, :, ::1] jnu_pec,
               double[:, :, ::1] jnpu_pec):
    cdef:
        int i, s, n, m, en
        int num_n_all = n_all.shape[0]
        double up, norm, jnup, jnpup
        cdouble h, u, v, jnu, jnpu, knv, knpv, a, b
        cdouble uc, vc, jnuc, jnpuc, knvc, knpvc, ac, bc
        cdouble ab[2]
        cdouble vals[2]
        cdouble val_u, val_v, ud, uod, vd, vod
    As_array = np.empty(num_n_all, dtype=complex)
    Bs_array = np.empty(num_n_all, dtype=complex)
    Ys_array = np.empty(num_n_all, dtype=complex)
    cdef:
        cdouble[:] As = As_array
        cdouble[:] Bs = Bs_array
        cdouble[:] Ys = Ys_array
    if creal(e2) < -1e6:
        for i in range(num_n_all):
            s = s_all[i]
            n = n_all[i]
            m = m_all[i]
            en = 1 if n == 0 else 2
            up = u_pec[s, n, m - 1]
            jnup = jnu_pec[s, n, m - 1]
            jnpup = jnpu_pec[s, n, m - 1]
            if s_all[i] == 0:
                norm = sqrt(M_PI * r * r / en  *
                             (1 - n * n / (up * up)) * jnup * jnup)
                As[i] = 1.0 / norm
                Bs[i] = 0.0
                Ys[i] = hs[i] / w
            else:
                norm = sqrt(M_PI * r * r / en * jnpup * jnpup)
                As[i] = 0.0
                Bs[i] = 1.0 / norm
                Ys[i] = e1 * w / hs[i]
        return As_array, Bs_array, Ys_array
    coefs_C(
        &hs[0], w, &s_all[0], &n_all[0], &m_all[0], num_n_all,
        r, e1, e2, &As[0], &Bs[0])
    for i in range(num_n_all):
        s = s_all[i]
        n = n_all[i]
        en = 1 if n == 0 else 2
        h = hs[i]
        u = (1 + 1j) * csqrt(-0.5j * (e1 * w * w - h * h)) * r
        jv_jvp(n, u, vals)
        jnu = vals[0]
        jnpu = vals[1]
        v = (1 - 1j) * csqrt(0.5j * (- e2 * w * w + h * h)) * r
        kv_kvp(n, v, vals)
        knv = vals[0]
        knpv = vals[1]
        val_u = 2 * M_PI * r * r / en
        val_v = val_u * ((u * jnu) / (v * knv)) ** 2
        ud = upart_diag(n, u, jnu, jnpu, u, jnu, jnpu)
        vd = vpart_diag(n, v, knv, knpv, v, knv, knpv)
        uod = upart_off(n, u, jnu, u, jnu)
        vod = vpart_off(n, v, knv, v, knv)
        a = As[i]
        b = Bs[i]
        Ys[i] = (val_u * (h / w * a * (a * ud + b * uod) +
                          e1 * w / h * b * (b * ud + a * uod)) -
                 val_v * (h / w * a * (a * vd + b * vod) +
                          e2 * w / h * b * (b * vd + a * vod)))
    return As_array, Bs_array, Ys_array


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def uvABY_cython(cdouble w, double r, long[::1] s_all, long[::1] n_all,
                 long[::1] m_all, cdouble[::1] hs, cdouble e1, cdouble e2,
                 double[:, :, ::1] u_pec, double[:, :, ::1] jnu_pec,
                 double[:, :, ::1] jnpu_pec):
    cdef:
        int i, s, n, m, en
        int num_n_all = n_all.shape[0]
        double up, norm, jnup, jnpup
        cdouble h, u, v, jnu, jnpu, knv, knpv, a, b
        cdouble uc, vc, jnuc, jnpuc, knvc, knpvc, ac, bc
        cdouble ab[2]
        cdouble vals[2]
        cdouble val_u, val_v, ud, uod, vd, vod
    us_array = np.empty(num_n_all, dtype=complex)
    vs_array = np.empty(num_n_all, dtype=complex)
    As_array = np.empty(num_n_all, dtype=complex)
    Bs_array = np.empty(num_n_all, dtype=complex)
    Ys_array = np.empty(num_n_all, dtype=complex)
    cdef:
        cdouble[:] us = us_array
        cdouble[:] vs = vs_array
        cdouble[:] As = As_array
        cdouble[:] Bs = Bs_array
        cdouble[:] Ys = Ys_array
    if creal(e2) < -1e6:
        for i in range(num_n_all):
            s = s_all[i]
            n = n_all[i]
            m = m_all[i]
            en = 1 if n == 0 else 2
            up = u_pec[s, n, m - 1]
            jnup = jnu_pec[s, n, m - 1]
            jnpup = jnpu_pec[s, n, m - 1]
            if s_all[i] == 0:
                norm = sqrt(M_PI * r * r / en  *
                             (1 - n * n / (up * up)) * jnup * jnup)
                As[i] = 1.0 / norm
                Bs[i] = 0.0
                Ys[i] = hs[i] / w
            else:
                norm = sqrt(M_PI * r * r / en * jnpup * jnpup)
                As[i] = 0.0
                Bs[i] = 1.0 / norm
                Ys[i] = e1 * w / hs[i]
            us[i] = up
            vs[i] = (1 - 1j) * csqrt(0.5j * (- e2 * w * w + hs[i] * hs[i])) * r
        return us_array, vs_array, As_array, Bs_array, Ys_array
    coefs_C(
        &hs[0], w, &s_all[0], &n_all[0], &m_all[0], num_n_all,
        r, e1, e2, &As[0], &Bs[0])
    for i in range(num_n_all):
        s = s_all[i]
        n = n_all[i]
        en = 1 if n == 0 else 2
        h = hs[i]
        u = (1 + 1j) * csqrt(-0.5j * (e1 * w * w - h * h)) * r
        jv_jvp(n, u, vals)
        jnu = vals[0]
        jnpu = vals[1]
        v = (1 - 1j) * csqrt(0.5j * (- e2 * w * w + h * h)) * r
        kv_kvp(n, v, vals)
        knv = vals[0]
        knpv = vals[1]
        val_u = 2 * M_PI * r * r / en
        val_v = val_u * ((u * jnu) / (v * knv)) ** 2
        ud = upart_diag(n, u, jnu, jnpu, u, jnu, jnpu)
        vd = vpart_diag(n, v, knv, knpv, v, knv, knpv)
        uod = upart_off(n, u, jnu, u, jnu)
        vod = vpart_off(n, v, knv, v, knv)
        a = As[i]
        b = Bs[i]
        Ys[i] = (val_u * (h / w * a * (a * ud + b * uod) +
                          e1 * w / h * b * (b * ud + a * uod)) -
                 val_v * (h / w * a * (a * vd + b * vod) +
                          e2 * w / h * b * (b * vd + a * vod)))
        us[i] = u
        vs[i] = v
    return us_array, vs_array, As_array, Bs_array, Ys_array


@cython.cdivision(True)
cdef cdouble u_func(cdouble h2, cdouble w, cdouble e1, double r):
    return (1 + 1j) * csqrt(-0.5j * (e1 * w ** 2 - h2)) * r


@cython.cdivision(True)
cdef cdouble v_func(cdouble h2, cdouble w, cdouble e2, double r):
    return (1 - 1j) * csqrt(0.5j * (-e2 * w ** 2 + h2)) * r


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def eig_eq_cython(cdouble h2, cdouble w, str pol, int n, cdouble e1, cdouble e2, double r):
    """Return the left value of the characteristic equation"""

    cdef:
        cdouble u = u_func(h2, w, e1, r)
        cdouble v = v_func(h2, w, e2, r)
        cdouble jus, jpus
        cdouble kvs, kpvs
        cdouble te, tm, val
        cdouble vals[2]

    jv_jvp(n, u, vals)
    jus = vals[0]
    jpus = vals[1]
    kv_kvp(n, v, vals)
    kvs = vals[0]
    kpvs = vals[1]
    te = jpus / u + kpvs * jus / (v * kvs)
    tm = e1 * jpus / u + e2 * kpvs * jus / (v * kvs)
    if n == 0:
        if pol == "M":
            val = tm
        else:
            val = te
    else:
        val = (
            tm * te - h2 * (n / w) ** 2 * ((1 / u ** 2 + 1 / v ** 2) * jus) ** 2
        )
    return val


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def jac_cython(double[::1] h2vec, cdouble w, str pol, int n, cdouble e1, cdouble e2, double r):
    """Return Jacobian of the characteristic equation"""

    cdef:
        cdouble h2 = h2vec[0] + 1j * h2vec[1]
        cdouble u = u_func(h2, w, e1, r)
        cdouble v = v_func(h2, w, e2, r)
        cdouble jus, jpus
        cdouble kvs, kpvs, k_k
        cdouble te, tm, val
        cdouble du_dh2, dv_dh2, dte_du, dte_dv, dtm_du, dtm_dv, dre_dh2
        cdouble vals[2]

    jv_jvp(n, u, vals)
    jus = vals[0]
    jpus = vals[1]
    kv_kvp(n, v, vals)
    kvs = vals[0]
    kpvs = vals[1]
    k_k = kpvs / kvs
    te = jpus / u + k_k * jus / v
    tm = e1 * jpus / u + e2 * k_k * jus / v

    du_dh2 = -r / (2 * u)
    dv_dh2 = r / (2 * v)

    dte_du = -(1 - n ** 2 / u ** 2) * jus / u - 2 * jpus / u ** 2 + k_k * jpus / v

    dte_dv = jus * (n ** 2 / v + v * (1 - k_k ** 2) - 2 * k_k) / v ** 2

    dtm_du = e1 * dte_du
    dtm_dv = e2 * dte_dv


    if n == 0:
        if pol == "M":
            val = dtm_du * du_dh2 + dtm_dv * dv_dh2
        else:
            val = dte_du * du_dh2 + dte_dv * dv_dh2
    else:
        dre_dh2 = (
            -((n / w) ** 2)
            * jus
            * (
                jus
                * (
                    (1 / u ** 2 + 1 / v ** 2) ** 2
                    - r * h2 * (1 / u ** 4 - 1 / v ** 4)
                )
                + jpus * 2 * h2 * (1 / u ** 2 + 1 / v ** 2) ** 2
            )
        )
        val = (
            (dte_du * du_dh2 + dte_dv * dv_dh2) * tm
            + (dtm_du * du_dh2 + dtm_dv * dv_dh2) * te
            + dre_dh2
        )
    return np.array([[val.real, -val.imag], [val.imag, val.real]])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def func_cython(
    double[::1] h2vec, cdouble w, str pol, int n, cdouble e1, cdouble e2, double r, cdouble[::1] roots
):
    """Return the value of the characteristic equation

    Args:
        h2: The square of the phase constant.
        w: The angular frequency
        pol: The polarization
        n: The order of the modes
        e1: The permittivity of the core
        e2: The permittivity of the clad.
        r: The radius of hole
        roots: Already obtained roots
    Returns:
        val: A complex indicating the left-hand value of the characteristic
            equation.
    """
    cdef:
        int i
        cdouble h2 = h2vec[0] + 1j * h2vec[1]
        cdouble u = u_func(h2, w, e1, r)
        cdouble v = v_func(h2, w, e2, r)
        cdouble jus, jpus
        cdouble kvs, kpvs
        cdouble te, tm
        cdouble vals[2]
        cdouble f
        cdouble denom = 1.0 + 0.0j
        double norm
        int num = len(roots)

    jv_jvp(n, u, vals)
    jus = vals[0]
    jpus = vals[1]
    kv_kvp(n, v, vals)
    kvs = vals[0]
    kpvs = vals[1]
    te = jpus / u + kpvs * jus / (v * kvs)
    tm = e1 * jpus / u + e2 * kpvs * jus / (v * kvs)
    if n == 0:
        if pol == "M":
            f = tm
        else:
            f = te
    else:
        f = (
            tm * te - h2 * (n / w) ** 2 * ((1 / u ** 2 + 1 / v ** 2) * jus) ** 2
        )
    for i in range(num):
        denom *= h2 - roots[i]
    norm = cabs(denom)
    if norm < 1e-14:
        denom /= norm * 1e14
    f /= denom
    return f.real, f.imag
