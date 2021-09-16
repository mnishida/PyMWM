# -*- coding: utf-8 -*-
# cython: profile=False
from . cimport M_PI, cabs, cdouble, creal, csqrt, sqrt
from .bessel_utils cimport jv, jv_jvp, jvp, kv, kv_kvp, kvp


cdef void jk_to_coefs(int n, int pol, cdouble h,
                      cdouble u, cdouble jnu, cdouble jnpu,
                      cdouble v, cdouble knv, cdouble knpv,
                      cdouble w, double r,
                      cdouble e1, cdouble e2,
                      cdouble ab[2]) nogil

cdef void coefs_pec_C(long *s_all, long *n_all, long  *m_all, int num_n_all,
              double r, cdouble *As, cdouble *Bs)

cdef void coefs_C(
    cdouble *hs, cdouble w, long *s_all, long *n_all, long  *m_all,
    int num_n_all, double r, cdouble e1, cdouble e2,
    cdouble *As, cdouble *Bs) nogil

cdef cdouble upart_diag(int n, cdouble uc, cdouble jnuc, cdouble jnpuc,
                        cdouble u, cdouble jnu, cdouble jnpu) nogil

cdef cdouble upart_off(int n, cdouble uc, cdouble jnuc, cdouble u, cdouble jnu) nogil

cdef cdouble vpart_diag(int n, cdouble vc, cdouble knvc, cdouble knpvc,
                        cdouble v, cdouble knv, cdouble knpv) nogil

cdef cdouble vpart_off(int n, cdouble vc, cdouble knvc, cdouble v, cdouble knv) nogil
