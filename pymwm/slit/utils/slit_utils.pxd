# -*- coding: utf-8 -*-
# cython: profile=False

ctypedef double complex cdouble

from libc.stdio cimport printf
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, sin, cos, atan2, M_PI

cdef extern from "complex.h" nogil:
    cdouble csqrt (cdouble z)
    double cabs (cdouble z)
    cdouble conj (cdouble z)
    cdouble cexp (cdouble z)
    cdouble csin (cdouble z)
    cdouble ccos (cdouble z)
    double creal (cdouble z)
    double cimag (cdouble z)
#     cdouble cpow(cdouble x, cdouble n)

cdef coefs_pec_C(long *s_all, long *n_all, int num_n_all,
              double r, double *As, double *Bs)

cdef cdouble csinc(cdouble x) nogil

cdef void coefs_C(
    cdouble *hs, cdouble w, long *s_all, long *n_all,
    int num_n_all, double r, cdouble e1, cdouble e2,
    double *As, double *Bs) nogil
