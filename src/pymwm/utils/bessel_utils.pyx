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
cdef cdouble jvp(int n, cdouble z) nogil:
    return 0.5 * (jv(n - 1, z) - jv(n + 1, z))

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
cdef void jv_jvp_jvpp(int n, cdouble z, cdouble vals[3]) nogil:
    cdef:
         cdouble j = jv(n, z)
         cdouble jp = -jv(n + 1, z) + n * j / z
    vals[0] = j
    vals[1] = jp
    vals[2] = -jp / z - (1 - n ** 2 / z ** 2) * j


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
cdef cdouble yvp(int n, cdouble z) nogil:
    return 0.5 * (yv(n - 1, z) - yv(n + 1, z))

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
cdef void yv_yvp_yvpp(int n, cdouble z, cdouble vals[3]) nogil:
    cdef:
         cdouble y = yv(n, z)
         cdouble yp = -yv(n + 1, z) + n * y / z
    vals[0] = y
    vals[1] = yp
    vals[2] = -yp / z - (1 - n ** 2 / z ** 2) * y


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double kn(int n, double z) nogil:
    return kv(n, z)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double knp(int n, double z) nogil:
    return -0.5 * (kn(n - 1, z) + kn(n + 1, z))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void kn_knp(int n, double z, double vals[2]) nogil:
    cdef:
        double k = kn(n, z)
    vals[0] = k
    vals[1] = -kn(n + 1, z) + n * k / z


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef cdouble kvp(int n, cdouble z) nogil:
    return -0.5 * (kv(n - 1, z) + kv(n + 1, z))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void kv_kvp(int n, cdouble z, cdouble vals[2]) nogil:
    cdef:
        cdouble k = kv(n, z)
    vals[0] = k
    vals[1] = -kv(n + 1, z) + n * k / z
