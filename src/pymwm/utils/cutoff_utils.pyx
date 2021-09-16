# -*- coding: utf-8 -*-
# cython: profile=False
cimport cython


@cython.cdivision(True)
def f_fprime_cython(double x, double r_ratio, int n, str pol):
    cdef:
        double v = x * r_ratio
        double jv, jpv, jppv, yv, ypv, yppv
        double jx, jpx, jppx, yx, ypx, yppx
        double vals[3]
        double f
        double fp

    if pol == "E":
        jn_jnp_jnpp(n, v, vals)
        jv = vals[0]
        jpv = vals[1]
        jppv = vals[2]

        yn_ynp_ynpp(n, v, vals)
        yv = vals[0]
        ypv = vals[1]
        yppv = vals[2]

        jn_jnp_jnpp(n, x, vals)
        jx = vals[0]
        jpx = vals[1]
        jppx = vals[2]

        yn_ynp_ynpp(n, x, vals)
        yx = vals[0]
        ypx = vals[1]
        yppx = vals[2]

        f = jpv * ypx - ypv * jpx
        fp = r_ratio * jppv * ypx + jpv * yppx - r_ratio * yppv * jpx - ypv * jppx
    else:
        jn_jnp(n, v, vals)
        jv = vals[0]
        jpv = vals[1]

        yn_ynp(n, v, vals)
        yv = vals[0]
        ypv = vals[1]

        jn_jnp(n, x, vals)
        jx = vals[0]
        jpx = vals[1]

        yn_ynp(n, x, vals)
        yx = vals[0]
        ypx = vals[1]

        f = jv * yx - yv * jx
        fp = r_ratio * jpv * yx + jv * ypx - r_ratio * ypv * jx - yv * jpx
    return f, fp


@cython.cdivision(True)
def f_cython(double x, double r_ratio, int n, str pol):
    cdef:
        double v = x * r_ratio

    if pol == "E":
        return jnp(n, v) * ynp(n, x) - ynp(n, v) * jnp(n, x)
    return  jn(n, v) * yn(n, x) - yn(n, v) * jn(n, x)


@cython.cdivision(True)
def fprime_cython(double x, double r_ratio, int n, str pol):
    cdef:
        double v = x * r_ratio
        double jv, jpv, jppv, yv, ypv, yppv
        double jx, jpx, jppx, yx, ypx, yppx
        double vals[3]

    jn_jnp_jnpp(n, v, vals)
    jv = vals[0]
    jpv = vals[1]
    jppv = vals[2]

    yn_ynp_ynpp(n, v, vals)
    yv = vals[0]
    ypv = vals[1]
    yppv = vals[2]

    jn_jnp_jnpp(n, x, vals)
    jx = vals[0]
    jpx = vals[1]
    jppx = vals[2]

    yn_ynp_ynpp(n, x, vals)
    yx = vals[0]
    ypx = vals[1]
    yppx = vals[2]

    if pol == "E":
        return r_ratio * jppv * ypx + jpv * yppx - r_ratio * yppv * jpx - ypv * jppx
    else:
        return r_ratio * jpv * yx + jv * ypx - r_ratio * ypv * jx - yv * jpx
