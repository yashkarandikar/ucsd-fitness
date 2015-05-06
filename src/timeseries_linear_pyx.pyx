import numpy as np
cimport numpy as np

DTYPE = np.float64 
ctypedef np.float64_t DTYPE_t

def F_pyx(np.ndarray[DTYPE_t, ndim=1] theta, np.ndarray[DTYPE_t, ndim=2] X, np.ndarray[DTYPE_t, ndim=1] y, double lam, only_f = False):
    cdef int N = X.shape[0]
    cdef int M = X.shape[1], i, k
    cdef double f = 0.0, err
    cdef np.ndarray[DTYPE_t, ndim=1] df = np.array([0.0] * M)
    for i in xrange(0, N):
        err = y[i] - X[i, :].dot(theta)
        f += err*err
        for k in xrange(0, M):
            df[k] -= 2 * err * X[i, k]

    f /= float(N)
    df /= float(N)

    # regularization
    for k in xrange(1, M):
        f += lam * theta[k] * theta[k]
        df[k] += 2 * lam * theta[k]

    if (only_f):
        return f
    else:
        return [f, df]

