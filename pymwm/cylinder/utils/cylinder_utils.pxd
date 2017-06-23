# -*- coding: utf-8 -*-
# cython: profile=False

import numpy as np
cimport numpy as np
ctypedef np.complex128_t cdouble;

from libc.stdio cimport printf
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, sin, cos, atan2, M_PI

cdef extern from "<complex>" namespace "std" nogil:
    cdouble csqrt "sqrt" (cdouble z)
    double cabs "abs" (cdouble z)
    cdouble cconj "conj" (cdouble z)
    cdouble cexp "exp" (cdouble z)
    double creal "real" (cdouble z)
    double cimag "imag" (cdouble z)
#     cdouble cpow "pow"(cdouble x, cdouble n)

cdef extern from "c_complex_bessel.h" nogil:
    cdouble c_besselJ(int n, cdouble z) nogil
    cdouble c_besselJp(int n, cdouble z) nogil
    cdouble c_besselK(int n, cdouble z) nogil
    cdouble c_besselKp(int n, cdouble z) nogil

cdef void jk_to_coefs(int n, int pol, cdouble h,
                      cdouble u, cdouble jnu, cdouble jnpu,
                      cdouble v, cdouble knv, cdouble knpv,
                      cdouble w, double r,
                      cdouble e1, cdouble e2,
                      cdouble ab[2]) nogil

cdef coefs_pec_C(long *s_all, long *n_all, long  *m_all, int num_n_all,
              double r, cdouble *As, cdouble *Bs)

cdef void coefs_C(
    cdouble *hs, cdouble w, long *s_all, long *n_all, long  *m_all,
    int num_n_all, double r, cdouble e1, cdouble e2,
    cdouble *As, cdouble *Bs) nogil

cdef cdouble upart_diag(int n, cdouble uc, cdouble jnuc, cdouble jnpuc,
                        cdouble u, cdouble jnu, cdouble jnpu) nogil

cdef inline cdouble upart_off(int n, cdouble uc, cdouble jnuc,
                              cdouble u, cdouble jnu) nogil

cdef cdouble vpart_diag(int n, cdouble vc, cdouble knvc, cdouble knpvc,
                        cdouble v, cdouble knv, cdouble knpv) nogil

cdef inline cdouble vpart_off(int n, cdouble vc, cdouble knvc,
                              cdouble v, cdouble knv) nogil
