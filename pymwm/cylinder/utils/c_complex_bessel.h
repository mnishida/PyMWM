#ifdef __cplusplus
#include <complex>
    typedef std::complex< double > cdouble;
#else
#include <complex.h>
    typedef double complex cdouble;
#endif

#ifdef __cplusplus
extern "C" {
#endif
    cdouble c_besselJ(int n, cdouble z);
    cdouble c_besselJp(int n, cdouble z);
    cdouble c_besselK(int n, cdouble z);
    cdouble c_besselKp(int n, cdouble z);
#ifdef __cplusplus
};
#endif
