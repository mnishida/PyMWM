import numpy as np

cimport cython
cimport numpy as np

ctypedef np.complex128_t cdouble

from libc.math cimport INFINITY, M_PI, atan2, cos, fabs, sin, sqrt
from libc.stdio cimport printf
from libc.stdlib cimport free, malloc


cdef extern from "<complex>" namespace "std"  nogil:
    cdouble csqrt "sqrt" (cdouble z)
    double cabs "abs" (cdouble z)
    cdouble cconj "conj" (cdouble z)
    cdouble csin "sin" (cdouble z)
    cdouble ccos "cos" (cdouble z)
    cdouble ctan "tan" (cdouble z)
    cdouble csinh "sinh" (cdouble z)
    cdouble ccosh "cosh" (cdouble z)
    cdouble ctanh "tanh" (cdouble z)
    cdouble cexp "exp" (cdouble z)
    double creal "real" (cdouble z)
    double cimag "imag" (cdouble z)
    cdouble cpow "pow"(cdouble x, cdouble n)
