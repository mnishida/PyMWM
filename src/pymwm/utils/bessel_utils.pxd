# -*- coding: utf-8 -*-
# cython: profile=False
from scipy.special.cython_special cimport iv, ive, jv, jve, kv, kve, yv, yve

from . cimport cdouble, cexp, cimag, creal, fabs


cdef double jn(int n, double z) nogil
cdef double jnp(int n, double z) nogil
cdef void jn_jnp(int n, double z, double vals[2]) nogil
cdef void jn_jnp_jnpp(int n, double z, double vals[3]) nogil
cdef void jn_jnp_jnpp_jnppp(int n, double z, double vals[4]) nogil

cdef double yn(int n, double z) nogil
cdef double ynp(int n, double z) nogil
cdef void yn_ynp(int n, double z, double vals[2]) nogil
cdef void yn_ynp_ynpp(int n, double z, double vals[3]) nogil
cdef void yn_ynp_ynpp_ynppp(int n, double z, double vals[4]) nogil

cdef void jv_jvp(int n, cdouble z, cdouble vals[2]) nogil
cdef void kv_kvp(int n, cdouble z, cdouble vals[2]) nogil

cdef void jve_jvpe_jvppe(int n, cdouble z, cdouble vals[3]) nogil
cdef void yve_yvpe_yvppe(int n, cdouble z, cdouble vals[3]) nogil
cdef void ive_ivpe_ivppe(int n, cdouble z, cdouble vals[3]) nogil
cdef void kve_kvpe_kvppe(int n, cdouble z, cdouble vals[3]) nogil
