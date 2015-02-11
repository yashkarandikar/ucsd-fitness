import numpy as np
cimport numpy as np
import time as time
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def E_pyx(np.ndarray[DTYPE_t, ndim=1] theta, np.ndarray[DTYPE_t, ndim=2] data, np.float64_t lam):
    # error function to be minimized
    # assumes data has 4 columns : user_id, user_number, distance, duration and that it is sorted
    cdef double t1 = time.time()
    cdef int i = 0
    cdef int N = data.shape[0]
    cdef double e = 0, alpha, d, t, temp
    cdef double alpha_all = theta[-3]
    cdef double theta_0 = theta[-2]
    cdef double theta_1 = theta[-1]
    cdef int u
    while i < N:
        u = int(data[i, 0])
        alpha = theta[u]
        while i < N and data[i, 0] == u:
            d = data[i, 2]
            t = data[i, 3]
            #e += math.pow(alpha * (theta_0 + theta_1 * d) - t, 2)
            temp = (alpha + alpha_all) * (theta_0 + theta_1 * d) - t 
            e += temp * temp
            i += 1
    # add regularization norm
    e += lam * theta[:-3].dot(theta[:-3])
    
    cdef double t2 = time.time()
    #print "E = %f, time taken = %f" % (e, t2 - t1)
    return e

def Eprime_pyx(np.ndarray[DTYPE_t, ndim=1] theta, np.ndarray[DTYPE_t, ndim=2] data, np.float64_t lam):
    from predictor_duration_user import get_user_count
    cdef double t1 = time.time()
    cdef int N = data.shape[0]
    #n_users = int(data[-1, 0]) + 1
    cdef int n_users = get_user_count(data)
    assert(theta.shape[0] == n_users + 3)
    cdef double alpha = theta[-3]
    cdef double theta_0 = theta[-2]
    cdef double theta_1 = theta[-1]
    #dE = np.array([0.0] * len(theta))
    dE = [0.0] * len(theta)
    cdef int i = 0
    cdef np.ndarray col0 = np.ravel(data[:, 0])
    cdef np.ndarray uins = np.array(range(0, n_users))
    u_indices = list(np.searchsorted(col0, uins))
    u_indices.append(N)
    
    cdef double dE_theta0 = 0.0
    cdef double dE_theta1 = 0.0

    cdef int start_u, end_u
    cdef double alpha_u, t, d, t0_t1_d, a_t0_t1_d, dE0

    for i in xrange(0, n_users):
        start_u = u_indices[i]
        end_u = u_indices[i+1]
        alpha_u = theta[i]
        for j in xrange(start_u, end_u):
            t = data[j, 3]
            d = data[j, 2]
            
            t0_t1_d = theta_0 + theta_1 * d

            tpred = (alpha_u + alpha) * t0_t1_d
            dE[i] += 2 * (tpred - t) * t0_t1_d 
            dE[-3] += 2 * (tpred - t) * t0_t1_d
            
            #a_t0_t1_d = alpha_u * t0_t1_d
            #dE[i] = dE[i] + 2 * (a_t0_t1_d - t) * t0_t1_d

            # dE / d_theta_0 and 1
            dE0 = 2 * (alpha_u + alpha) * (tpred - t)
            dE_theta0 += dE0
            dE_theta1 += dE0 * d
        
    dE[-2] = dE_theta0
    dE[-1] = dE_theta1

    # regularization
    #dE = dE + lam * np.multiply(dE, (2 * theta))
    for u in xrange(0, n_users):
        dE[u] += 2 * lam * theta[u]

    cdef double t2 = time.time()
    #print "E prime : time taken = ", t2 - t1
    return np.array(dE)
    #return dE

