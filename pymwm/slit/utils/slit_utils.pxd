# -*- coding: utf-8 -*-
# cython: profile=False
import numpy as np

cimport numpy as np

ctypedef np.complex128_t cdouble;

from libc.math cimport M_PI, atan2, cos, sin, sqrt
from libc.stdio cimport printf
from libc.stdlib cimport free, malloc


cdef extern from "<complex>" namespace "std"  nogil:
    cdouble csqrt "sqrt" (cdouble z)
    double cabs "abs" (cdouble z)
    cdouble cconj "conj" (cdouble z)
    cdouble csin "sin" (cdouble z)
    cdouble ccos "cos" (cdouble z)
    cdouble cexp "exp" (cdouble z)
    double creal "real" (cdouble z)
    double cimag "imag" (cdouble z)
#     cdouble cpow "pow"(cdouble x, cdouble n)

cdef coefs_pec_C(long *s_all, long *n_all, int num_n_all,
              double r, cdouble *As, cdouble *Bs)

cdef cdouble csinc(cdouble x) nogil

cdef void coefs_C(
    cdouble *hs, cdouble w, long *s_all, long *n_all,
    int num_n_all, double r, cdouble e1, cdouble e2,
    cdouble *As, cdouble *Bs) nogil
