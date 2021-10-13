# -*- coding: utf-8 -*-
# cython: profile=False
from . cimport cabs, cdouble, cimag, creal, csqrt


cdef cdouble det_cython(cdouble[:, ::1] a)
cdef cdouble det4(cdouble[:, ::1] a) nogil
cdef cdouble deriv_det4(cdouble[:, ::1] a, cdouble[:, ::] b) nogil
cdef cdouble deriv_det2(cdouble[:, ::1] a, cdouble[:, ::] b)
