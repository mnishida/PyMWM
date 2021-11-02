# -*- coding: utf-8 -*-
# cython: profile=False
from . cimport cabs, cdouble, cimag, creal, csqrt, printf


cdef cdouble det_cython(cdouble[:, ::1] a)
cdef cdouble det4(cdouble[:, ::1] a) nogil
cdef cdouble deriv_det4(cdouble[:, ::1] a, cdouble[:, ::] b) nogil
cdef cdouble deriv_det2(cdouble[:, ::1] a, cdouble[:, ::] b)
cdef void solve(cdouble a[3][3], cdouble b[3], cdouble x[3]) nogil
