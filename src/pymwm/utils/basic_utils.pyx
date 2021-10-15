# -*- coding: utf-8 -*-
# cython: profile=False
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef cdouble ctanh(cdouble z) nogil:
    cdef cdouble val
    if creal(z) >= 0:
        val = cexp(-2 * z)
        return (1 - val) / (1 + val)
    else:
        val = cexp(2 * z)
        return (val - 1) / (val + 1)
