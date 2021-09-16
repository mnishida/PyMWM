# -*- coding: utf-8 -*-
# cython: profile=False
from scipy.special.cython_special cimport jv, kv, yv

from . cimport cdouble


cdef double jn(int n, double z) nogil
cdef double jnp(int n, double z) nogil
cdef void jn_jnp(int n, double z, double vals[2]) nogil
cdef void jn_jnp_jnpp(int n, double z, double vals[3]) nogil

cdef cdouble jvp(int n, cdouble z) nogil
cdef void jv_jvp(int n, cdouble z, cdouble vals[2]) nogil
cdef void jv_jvp_jvpp(int n, cdouble z, cdouble vals[3]) nogil

cdef double yn(int n, double z) nogil
cdef double ynp(int n, double z) nogil
cdef void yn_ynp(int n, double z, double vals[2]) nogil
cdef void yn_ynp_ynpp(int n, double z, double vals[3]) nogil

cdef cdouble yvp(int n, cdouble z) nogil
cdef void yv_yvp(int n, cdouble z, cdouble vals[2]) nogil
cdef void yv_yvp_yvpp(int n, cdouble z, cdouble vals[3]) nogil

cdef double knp(int n, double z) nogil
cdef void kn_knp(int n, double z, double vals[2]) nogil

cdef cdouble kvp(int n, cdouble z) nogil
cdef void kv_kvp(int n, cdouble z, cdouble vals[2]) nogil
