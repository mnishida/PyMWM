#include <complex_bessel.h>
#include "c_complex_bessel.h"

using namespace std;

cdouble c_besselJ(int n, cdouble z) {
    return sp_bessel::besselJ(n, z);
}

cdouble c_besselJp(int n, cdouble z) {
    return sp_bessel::besselJp(n, z);
}

cdouble c_besselK(int n, cdouble z) {
    return sp_bessel::besselK(n, z);
}

cdouble c_besselKp(int n, cdouble z) {
    return sp_bessel::besselKp(n, z);
}

