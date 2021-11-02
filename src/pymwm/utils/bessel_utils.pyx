# -*- coding: utf-8 -*-
# cython: profile=False
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double jn(int n, double z) nogil:
    return jv(n, z)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double jnp(int n, double z) nogil:
    return 0.5 * (jv(n - 1, z) - jv(n + 1, z))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void jn_jnp(int n, double z, double vals[2]) nogil:
    cdef:
         double j = jn(n, z)
    vals[0] = j
    vals[1] = -jn(n + 1, z) + n * j / z

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void jn_jnp_jnpp(int n, double z, double vals[3]) nogil:
    cdef:
         double j = jn(n, z)
         double jp = -jn(n + 1, z) + n * j / z
    vals[0] = j
    vals[1] = jp
    vals[2] = -jp / z - (1 - n ** 2 / z ** 2) * j

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void jn_jnp_jnpp_jnppp(int n, double z, double vals[4]) nogil:
    cdef:
         double j = jn(n, z)
         double jp = -jn(n + 1, z) + n * j / z
         double jpp = -jp / z - (1 - n ** 2 / z ** 2) * j
    vals[0] = j
    vals[1] = jp
    vals[2] = jpp
    vals[3] = -jpp / z - jp + (n ** 2 + 1) * jp / z ** 2 - 2 * n ** 2 / z ** 3 * j


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double yn(int n, double z) nogil:
    return yv(n, z)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double ynp(int n, double z) nogil:
    return 0.5 * (yn(n - 1, z) - yn(n + 1, z))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void yn_ynp(int n, double z, double vals[2]) nogil:
    cdef:
         double y = yn(n, z)
    vals[0] = y
    vals[1] = -yn(n + 1, z) + n * y / z

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void yn_ynp_ynpp(int n, double z, double vals[3]) nogil:
    cdef:
         double y = yn(n, z)
         double yp = -yn(n + 1, z) + n * y / z
    vals[0] = y
    vals[1] = yp
    vals[2] = -yp / z - (1 - n ** 2 / z ** 2) * y

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void yn_ynp_ynpp_ynppp(int n, double z, double vals[4]) nogil:
    cdef:
         double y = yn(n, z)
         double yp = -yn(n + 1, z) + n * y / z
         double ypp = -yp / z - (1 - n ** 2 / z ** 2) * y
    vals[0] = y
    vals[1] = yp
    vals[2] = ypp
    vals[3] = -ypp / z - yp + (n ** 2 + 1) * yp / z ** 2 - 2 * n ** 2 / z ** 3 * y


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void jv_jvp(int n, cdouble z, cdouble vals[2]) nogil:
    cdef:
         cdouble j = jv(n, z)
    vals[0] = j
    vals[1] = -jv(n + 1, z) + n * j / z


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void yv_yvp(int n, cdouble z, cdouble vals[2]) nogil:
    cdef:
         cdouble y = yv(n, z)
    vals[0] = y
    vals[1] = -yv(n + 1, z) + n * y / z


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def yv_yvp_cython(int n, cdouble z):
    cdef cdouble vals[2]
    yv_yvp(n, z, vals)
    return vals[0], vals[1]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void kv_kvp(int n, cdouble z, cdouble vals[2]) nogil:
    cdef:
        cdouble k = kv(n, z)
    vals[0] = k
    vals[1] = -kv(n + 1, z) + n * k / z


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void iv_ivp(int n, cdouble z, cdouble vals[2]) nogil:
    cdef:
        cdouble i = iv(n, z)
    vals[0] = i
    vals[1] = iv(n + 1, z) + n * i / z


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void jve_jvpe_jvppe(int n, cdouble z, cdouble vals[3]) nogil:
    """ jv * exp(1j * z), jvp * exp(1j * z), jvpp * exp(1j * z) """
    cdef int sign
    if cimag(z) > 0:
        sign = 1
    elif cimag(z) == 0 :
        sign = 0
    else:
        sign = -1
    cdef:
        cdouble ph = cexp(1j * sign * creal(z))
        cdouble j = jve(n, z) * ph
        cdouble jp = -jve(n + 1, z) * ph + n * j / z
    vals[0] = j
    vals[1] = jp
    vals[2] = -jp / z - (1 - n ** 2 / z ** 2) * j


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void yve_yvpe_yvppe(int n, cdouble z, cdouble vals[3]) nogil:
    """ yv * exp(1j * z), yvp * exp(1j * z), yvpp * exp(1j * z) """
    cdef int sign
    if cimag(z) > 0:
        sign = 1
    elif cimag(z) == 0 :
        sign = 0
    else:
        sign = -1
    cdef:
        cdouble ph = cexp(1j * sign * creal(z))
        cdouble y = yve(n, z) * ph
        cdouble yp = -yve(n + 1, z) * ph + n * y / z
    vals[0] = y
    vals[1] = yp
    vals[2] = -yp / z - (1 - n ** 2 / z ** 2) * y


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void ive_ivpe_ivppe(int n, cdouble z, cdouble vals[3]) nogil:
    """ iv * exp(-z), ivp * exp(-z), ivpp * exp(-z) """
    cdef int sign
    if creal(z) > 0:
        sign = 1
    elif creal(z) == 0 :
        sign = 0
    else:
        sign = -1
    cdef:
        cdouble ph = cexp(-1j * sign * cimag(z))
        cdouble i = ive(n, z) * ph
        cdouble ip = ive(n + 1, z) * ph + n * i / z
    vals[0] = i
    vals[1] = ip
    vals[2] = -ip / z + (1 + n ** 2 / z ** 2) * i


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void kve_kvpe_kvppe(int n, cdouble z, cdouble vals[3]) nogil:
    """ kv * exp(z), kvp * exp(z), kvpp * exp(z) """
    cdef:
        cdouble k = kve(n, z)
        cdouble kp = -kve(n + 1, z) + n * k / z
    vals[0] = k
    vals[1] = kp
    vals[2] = -kp / z + (1 + n ** 2 / z ** 2) * k
