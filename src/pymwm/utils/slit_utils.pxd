# -*- coding: utf-8 -*-
# cython: profile=False
from . cimport cabs, ccos, cdouble, creal, csin, csqrt, ctan, sqrt


cdef coefs_pec_C(long *s_all, long *n_all, int num_n_all,
              double r, cdouble *As, cdouble *Bs)

cdef cdouble csinc(cdouble x) nogil

cdef void coefs_C(
    cdouble *hs, cdouble w, long *s_all, long *n_all,
    int num_n_all, double r, cdouble e1, cdouble e2,
    cdouble *As, cdouble *Bs) nogil
