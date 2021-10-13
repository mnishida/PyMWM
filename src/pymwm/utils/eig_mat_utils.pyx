# -*- coding: utf-8 -*-
# cython: profile=False
import numpy as np

cimport cython
cimport numpy as np

cdef:
    int i, j
    int inds4[24][4]
    int sign4[24]

sign4[0] = 1
inds4[0][0] = 0
inds4[0][1] = 1
inds4[0][2] = 2
inds4[0][3] = 3
sign4[1] = 1
inds4[1][0] = 0
inds4[1][1] = 3
inds4[1][2] = 1
inds4[1][3] = 2
sign4[2] = 1
inds4[2][0] = 0
inds4[2][1] = 2
inds4[2][2] = 3
inds4[2][3] = 1
sign4[3] = -1
inds4[3][0] = 0
inds4[3][1] = 3
inds4[3][2] = 2
inds4[3][3] = 1
sign4[4] = -1
inds4[4][0] = 0
inds4[4][1] = 1
inds4[4][2] = 3
inds4[4][3] = 2
sign4[5] = -1
inds4[5][0] = 0
inds4[5][1] = 2
inds4[5][2] = 1
inds4[5][3] = 3
sign4[6] = -1
inds4[6][0] = 1
inds4[6][1] = 0
inds4[6][2] = 2
inds4[6][3] = 3
sign4[7] = -1
inds4[7][0] = 1
inds4[7][1] = 3
inds4[7][2] = 0
inds4[7][3] = 2
sign4[8] = -1
inds4[8][0] = 1
inds4[8][1] = 2
inds4[8][2] = 3
inds4[8][3] = 0
sign4[9] = 1
inds4[9][0] = 1
inds4[9][1] = 3
inds4[9][2] = 2
inds4[9][3] = 0
sign4[10] = 1
inds4[10][0] = 1
inds4[10][1] = 0
inds4[10][2] = 3
inds4[10][3] = 2
sign4[11] = 1
inds4[11][0] = 1
inds4[11][1] = 2
inds4[11][2] = 0
inds4[11][3] = 3
sign4[12] = 1
inds4[12][0] = 2
inds4[12][1] = 0
inds4[12][2] = 1
inds4[12][3] = 3
sign4[13] = 1
inds4[13][0] = 2
inds4[13][1] = 3
inds4[13][2] = 0
inds4[13][3] = 1
sign4[14] = 1
inds4[14][0] = 2
inds4[14][1] = 1
inds4[14][2] = 3
inds4[14][3] = 0
sign4[15] = -1
inds4[15][0] = 2
inds4[15][1] = 3
inds4[15][2] = 1
inds4[15][3] = 0
sign4[16] = -1
inds4[16][0] = 2
inds4[16][1] = 0
inds4[16][2] = 3
inds4[16][3] = 1
sign4[17] = -1
inds4[17][0] = 2
inds4[17][1] = 1
inds4[17][2] = 0
inds4[17][3] = 3
sign4[18] = -1
inds4[18][0] = 3
inds4[18][1] = 0
inds4[18][2] = 1
inds4[18][3] = 2
sign4[19] = -1
inds4[19][0] = 3
inds4[19][1] = 2
inds4[19][2] = 0
inds4[19][3] = 1
sign4[20] = -1
inds4[20][0] = 3
inds4[20][1] = 1
inds4[20][2] = 2
inds4[20][3] = 0
sign4[21] = 1
inds4[21][0] = 3
inds4[21][1] = 2
inds4[21][2] = 1
inds4[21][3] = 0
sign4[22] = 1
inds4[22][0] = 3
inds4[22][1] = 0
inds4[22][2] = 2
inds4[22][3] = 1
sign4[23] = 1
inds4[23][0] = 3
inds4[23][1] = 1
inds4[23][2] = 0
inds4[23][3] = 2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef cdouble det4(cdouble[:, ::1] a) nogil:
    cdef:
        int i, j, n
        cdouble val = 0.0j
        cdouble elem
    for n in range(24):
        elem = sign4[n]
        for i in range(4):
            j = inds4[n][i]
            elem *= a[i, j]
        val += elem
    return val


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef cdouble deriv_det4(cdouble[:, ::1] a, cdouble[:, ::] b) nogil:
    cdef:
        int i, j, n, m
        cdouble val = 0.0j
        cdouble elem
    for n in range(24):
        for m in range(4):
            elem = sign4[n]
            for i in range(4):
                j = inds4[n][i]
                if j == m:
                    elem *= b[i, j]
                else:
                    elem *= a[i, j]
            val += elem
    return val


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef det2(cdouble[:, ::1] a):
    return a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef cdouble deriv_det2(cdouble[:, ::1] a, cdouble[:, ::] b):
    return b[0, 0] * a[1, 1] + a[0, 0] * b[1, 1] - b[0, 1] * a[1, 0] - a[0, 1] * b[1, 0]


def det4_cython(cdouble[:, ::1] a):
    return det4(a)


def deriv_det4_cython(cdouble[:, ::1] a, cdouble[:, ::1] b):
    return deriv_det4(a, b)


def deriv_det2_cython(cdouble[:, ::1] a, cdouble[:, ::1] b):
    return deriv_det2(a, b)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline cdouble det_cython(cdouble[:, ::1] a):
    return (
        a[0, 0] * (
            a[1, 1] * a[2, 2] * a[3, 3] + a[1, 3] * a[2, 1] * a[3, 2] + a[1, 2] * a[2, 3] * a[3, 1] -
            a[1, 3] * a[2, 2] * a[3, 1] - a[1, 1] * a[2, 3] * a[3, 2] - a[1, 2] * a[2, 1] * a[3, 3]) -
        a[0, 1] * (
            a[1, 0] * a[2, 2] * a[3, 3] + a[1, 3] * a[2, 0] * a[3, 2] + a[1, 2] * a[2, 3] * a[3, 0] -
            a[1, 3] * a[2, 2] * a[3, 0] - a[1, 0] * a[2, 3] * a[3, 2] - a[1, 2] * a[2, 0] * a[3, 3]) +
        a[0, 2] * (
            a[1, 0] * a[2, 1] * a[3, 3] + a[1, 3] * a[2, 0] * a[3, 1] + a[1, 1] * a[2, 3] * a[3, 0] -
            a[1, 3] * a[2, 1] * a[3, 0] - a[1, 0] * a[2, 3] * a[3, 1] - a[1, 1] * a[2, 0] * a[3, 3]) -
        a[0, 3] * (
            a[1, 0] * a[2, 1] * a[3, 2] + a[1, 2] * a[2, 0] * a[3, 1] + a[1, 1] * a[2, 2] * a[3, 0] -
            a[1, 2] * a[2, 1] * a[3, 0] - a[1, 0] * a[2, 2] * a[3, 1] - a[1, 1] * a[2, 0] * a[3, 2])
    )


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def solve_cython(cdouble[:, ::1] a, cdouble[::1] b):
    cdef cdouble A00, A01, A10, A11, B0, B1, inv_det, x, y
    A00 = a[0, 0] * a[1, 2] - a[1, 0] * a[0, 2]
    A01 = a[0, 1] * a[1, 2] - a[1, 1] * a[0, 2]
    A10 = a[1, 0] * a[2, 2] - a[2, 0] * a[1, 2]
    A11 = a[1, 1] * a[2, 2] - a[2, 1] * a[1, 2]
    B0 = b[0] * a[1, 2] - b[1] * a[0, 2]
    B1 = b[1] * a[2, 2] - b[2] * a[1, 2]
    inv_det = 1 / (A00 * A11 - A01 * A10)
    x = (B0 * A11 - B1 * A01) * inv_det
    y = (B1 * A00 - B0 * A10) * inv_det
    return  (
        x,
        y,
        (b[2] - a[2, 0] * x - a[2, 1] * y) / a[2, 2]
    )
