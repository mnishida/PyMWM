# -*- coding: utf-8 -*-
# cython: profile=False
import numpy as np
cimport numpy as np
cimport cython
cimport scipy.linalg.cython_blas as blas


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
        cdouble u2, jnu2, jnpu2
    u2 = u * u
    jnu2 = jnu * jnu
    jnpu2 = jnpu * jnpu
    if cabs(uc - u) < 1e-16:
        return jnu * jnpu / u + (jnpu2 + (1.0 - n2 / u2) * jnu2) / 2.0
    if cabs(uc + u) < 1e-16:
        return ((1 - ((n - 1) % 2) * 2) *
                (jnu * jnpu / u + (jnpu2 + (1.0 - n2 / u2) * jnu2) / 2.0))
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
        cdouble v2, knv2, knpv2
    v2 = v * v
    knv2 = knv * knv
    knpv2 = knpv * knpv
    if cabs(vc - v) < 1e-16:
        return (knv * knpv / v +
		(knpv2 - (1.0 + n2 / v2) * knv2) / 2.0)
    if cabs(vc + v) < 1e-16:
        return (1 - ((n - 1) % 2) * 2) * (
            knv * knpv / v +
	    (knpv2 - (1.0 + n2 / v2) * knv2) / 2.0)
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
    As_array = np.empty(hole.num_n_all, dtype=np.complex)
    Bs_array = np.empty(hole.num_n_all, dtype=np.complex)
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
cdef coefs_pec_C(long *s_all, long *n_all, long  *m_all, int num_n_all,
              double r, cdouble *As, cdouble *Bs):
    cdef:
        int i, n, m, en
        double u, norm, jnu, jnpu
    from scipy.special import jv, jvp, jn_zeros, jnp_zeros
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
        double norm
        cdouble uc, vc, jnuc, jnpuc, knvc, knpvc, ac, bc
        cdouble h, u, v, jnu, jnpu, knv, knpv, a, b
        cdouble ab[2]
        cdouble val_u, val_v, ud, uod, vd, vod
    for i in range(num_n_all):
        n = n_all[i]
        en = 1 if n == 0 else 2
        h = hs[i]
        s = s_all[i]
        u = csqrt(e1 * w * w - h * h) * r
        jnu = c_besselJ(n, u)
        jnpu = c_besselJp(n, u)
        v = (1 - 1j) * csqrt(0.5j * (- e2 * w * w + h * h)) * r
        knv = c_besselK(n, v)
        knpv = c_besselKp(n, v)
        uc = u.conjugate()
        jnuc = jnu.conjugate()
        jnpuc = jnpu.conjugate()
        vc = v.conjugate()
        knvc = knv.conjugate()
        knpvc = knpv.conjugate()        
        jk_to_coefs(n, s, h, u, jnu, jnpu, v, knv, knpv, w, r, e1, e2, ab)
        a = ab[0]
        b = ab[1]
        ac = a.conjugate()
        bc = b.conjugate()
        val_u = 2 * M_PI * r * r / en
        val_v = cabs((u * jnu) / (v * knv))
        val_v *= val_u * val_v
        ud = upart_diag(n, uc, jnuc, jnpuc, u, jnu, jnpu)
        vd = vpart_diag(n, vc, knvc, knpvc, v, knv, knpv)
        uod = upart_off(n, uc, jnuc, u, jnu)
        vod = vpart_off(n, vc, knvc, v, knv)
        norm = sqrt(
            creal(val_u * (
                a * (ac * ud + bc * uod) + b * (bc * ud + ac * uod)) -
             val_v * (
                 a * (ac * vd + bc * vod) + b * (bc * vd + ac * vod))
            ))
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
        cdouble val_u, val_v, ud, uod, vd, vod
    As_array = np.empty(num_n_all, dtype=np.complex)
    Bs_array = np.empty(num_n_all, dtype=np.complex)
    Ys_array = np.empty(num_n_all, dtype=np.complex)
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
        u = csqrt(e1 * w * w - h * h) * r
        jnu = c_besselJ(n, u)
        jnpu = c_besselJp(n, u)
        v = (1 - 1j) * csqrt(0.5j * (- e2 * w * w + h * h)) * r
        knv = c_besselK(n, v)
        knpv = c_besselKp(n, v)
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
        cdouble val_u, val_v, ud, uod, vd, vod
    us_array = np.empty(num_n_all, dtype=np.complex)
    vs_array = np.empty(num_n_all, dtype=np.complex)
    As_array = np.empty(num_n_all, dtype=np.complex)
    Bs_array = np.empty(num_n_all, dtype=np.complex)
    Ys_array = np.empty(num_n_all, dtype=np.complex)
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
        u = csqrt(e1 * w * w - h * h) * r
        jnu = c_besselJ(n, u)
        jnpu = c_besselJp(n, u)
        v = (1 - 1j) * csqrt(0.5j * (- e2 * w * w + h * h)) * r
        knv = c_besselK(n, v)
        knpv = c_besselKp(n, v)
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
