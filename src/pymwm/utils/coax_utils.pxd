# -*- coding: utf-8 -*-
# cython: profile=False
from . cimport INFINITY, M_PI, cabs, cdouble, cimag, creal, csqrt, log10, printf, sqrt
from .bessel_utils cimport (
    iv_ivp,
    ive,
    ive_ivpe_ivppe,
    ive_ivpe,
    jv_jvp,
    jve,
    jve_jvpe_jvppe,
    jve_jvpe,
    kv_kvp,
    kve,
    kve_kvpe_kvppe,
    kve_kvpe,
    yv_yvp,
    yve,
    yve_yvpe_yvppe,
    yve_yvpe,
)
from .eig_mat_utils cimport deriv_det2, deriv_det4, det4, solve


cdef cdouble u_func(cdouble h2, cdouble w, cdouble e1, double r) nogil
cdef cdouble v_func(cdouble h2, cdouble w, cdouble e1, double r) nogil
cdef void eig_mat_with_deriv(
    cdouble h2, cdouble w, str pol, int n, cdouble e1, cdouble e2, double r, double ri, cdouble[:, ::1] a, cdouble[:, ::1] b
)
cdef void eig_mat(
    cdouble h2, cdouble w, str pol, int n, cdouble e1, cdouble e2, double r, double ri, cdouble[:, ::1] a
)
cdef void coefs_C(
    cdouble *hs, cdouble w, long *s_all, long *n_all,
    int num_n_all, double r, double ri, cdouble e1, cdouble e2,
    cdouble *xs, cdouble *ys, cdouble *us, cdouble *vs,
    cdouble *jxs, cdouble *jpxs, cdouble *yxs, cdouble *ypxs, cdouble *iys, cdouble *ipys,
    cdouble *jus, cdouble *jpus, cdouble *yus, cdouble *ypus, cdouble *kvs, cdouble *kpvs,
    cdouble *A1s, cdouble *B1s, cdouble *A2s, cdouble *B2s,
    cdouble *C2s, cdouble *D2s, cdouble *A3s, cdouble *B3s,
    cdouble *Ys) nogil
