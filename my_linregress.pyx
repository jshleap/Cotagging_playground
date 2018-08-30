import numpy as np
from scipy.stats import distributions
cimport cython
#cimport numpy as np

#np.import_array()

@cython.cdivision(True)
@cython.boundscheck(False)
cpdef cy_linregress(x, y=None):
    cdef double TINY = 1.0e-20
    cdef double ssxm, ssxym, ssyxm, ssym, r_num, r_den, r, prob, t, r_t
    cdef int n, df
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    # average sum of squares:
    ssxm, ssxym, ssyxm, ssym = np.cov(x, y, bias=1).flat
    r_num = ssxym
    r_den = np.sqrt(ssxm * ssym)
    if r_den == 0.0:
        r = 0.0
    else:
        r = r_num / r_den
        # test for numerical error propagation
        if r > 1.0:
            r = 1.0
        elif r < -1.0:
            r = -1.0
    df = n - 2
    slope = r_num / ssxm
    r_t = r + TINY
    t = r * np.sqrt(df / ((1.0 - r_t)*(1.0 + r_t)))
    prob = 2 * distributions.t.sf(np.abs(t), df)

    return slope, r**2, prob

@jit
def nu_linregress(x, y):
    TINY = 1.0e-20
    x = np.asarray(x)
    y = np.asarray(y)
    arr = np.array([x, y])
    n = len(x)
    # average sum of squares:
    ssxm, ssxym, ssyxm, ssym = (np.dot(arr,arr.T)/n).flat
    r_num = ssxym
    r_den = np.sqrt(ssxm * ssym)
    if r_den == 0.0:
        r = 0.0
    else:
        r = r_num / r_den
        # test for numerical error propagation
        if r > 1.0:
            r = 1.0
        elif r < -1.0:
            r = -1.0
    df = n - 2
    slope = r_num / ssxm
    r_t = r + TINY
    t = r * np.sqrt(df / ((1.0 - r_t)*(1.0 + r_t)))
    prob = 2 * distributions.t.sf(np.abs(t), df)

    return slope, r**2, prob

