# -*- coding: utf-8 -*-
# cython: profile=False
from . cimport cabs, cdouble, cimag, creal, csqrt
from .bessel_utils cimport (
    ive,
    ive_ivpe_ivppe,
    jve,
    jve_jvpe_jvppe,
    kve,
    kve_kvpe_kvppe,
    yve,
    yve_yvpe_yvppe,
)
from .eig_mat_utils cimport deriv_det2, deriv_det4, det4


cdef cdouble u_func(cdouble h2, cdouble w, cdouble e1, double r)
cdef cdouble v_func(cdouble h2, cdouble w, cdouble e1, double r)
cdef void eig_mat_with_deriv(
    cdouble h2, cdouble w, str pol, int n, cdouble e1, cdouble e2, double r, double ri, cdouble[:, ::1] a, cdouble[:, ::1] b
)
cdef void eig_mat(
    cdouble h2, cdouble w, str pol, int n, cdouble e1, cdouble e2, double r, double ri, cdouble[:, ::1] a
)
