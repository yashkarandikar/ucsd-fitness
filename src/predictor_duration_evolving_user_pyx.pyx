import numpy as np
cimport numpy as np
import time as time
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


def F_pyx(np.ndarray[DTYPE_t, ndim=1] theta, np.ndarray[DTYPE_t, ndim=2] data, np.float64_t lam, np.int_t E, sigma):
    from predictor_duration_evolving_user import get_theta_0, get_theta_1, get_alpha_e, get_alpha_ue, get_user_count
    # error function to be minimized
    # assumes data has 4 columns : user_id, user_number, distance, duration and that it is sorted
    # theta - first UxE elements are per-user per-experience alpha values, next E elements are per experience offset alphas, last 2 are theta0 and theta1
    # sigma - set of experience levels for all workouts for all users.. sigma is a matrix.. sigma(u,i) = e_ui i.e experience level of user u at workout i - these values are NOT optimized by L-BFGS.. they are optimized by DP procedure

    cdef double t1 = time.time()
    cdef int U = get_user_count(data)
    assert(theta.shape[0] == U * E + E + 2)
    cdef int w = 0, i, u, e
    cdef int N = data.shape[0]
    cdef double f = 0
    cdef double theta_0 = get_theta_0(theta)
    cdef double theta_1 = get_theta_1(theta)
    cdef double a_ue, a_e, d, t, diff
    while w < N:    # over all workouts i.e. all rows in data
        u = int(data[w, 0])
        i = 0   # ith workout of user u
        while w < N and data[w, 0] == u:
            #e = sigma[u, i]
            e = sigma[u][i]
            a_ue = get_alpha_ue(theta, u, e, E)[0]
            a_e = get_alpha_e(theta, e, E, U)[0]
            d = data[w, 2]
            t = data[w, 3]
            diff = (a_e + a_ue) * (theta_0 + theta_1*d) - t
            f += diff * diff
            w += 1
            i += 1

    # add regularization norm
    cdef double reg = 0, a_i, a_i_plus_1
    for i in range(0, E - 1):
        a_i = get_alpha_e(theta, i, E, U)[0]
        a_i_plus_1 = get_alpha_e(theta, i + 1, E, U)[0]
        diff = a_i - a_i_plus_1
        reg += diff * diff
        for u in range(0, U):
            diff = get_alpha_ue(theta, u, i, E)[0] - get_alpha_ue(theta, u, i+1, E)[0]
            reg += diff * diff
    f += lam * reg
    
    cdef double t2 = time.time()
    #print "F = %f, time taken = %f" % (f, t2 - t1)
    return f

def Fprime_pyx(np.ndarray[DTYPE_t, ndim=1] theta, np.ndarray[DTYPE_t, ndim=2] data, np.float64_t lam, np.int_t E, sigma):
    from predictor_duration_evolving_user import get_theta_0, get_theta_1, get_alpha_e, get_alpha_ue, get_user_count
    # theta - first UxE elements are per-user per-experience alpha values, next E elements are per experience offset alphas, last 2 are theta0 and theta1
    # sigma - set of experience levels for all workouts for all users.. sigma is a matrix.. sigma(u,i) = e_ui i.e experience level of user u at workout i - these values are NOT optimized by L-BFGS.. they are optimized by DP procedure
    cdef double t1 = time.time()
    cdef int N = data.shape[0]
    cdef int U = get_user_count(data)
    assert(theta.shape[0] == U * E + E + 2)
    cdef double theta_0 = get_theta_0(theta)
    cdef double theta_1 = get_theta_1(theta)

    cdef np.ndarray[DTYPE_t, ndim=1] dE = np.array([0.0] * theta.shape[0])

    cdef int w = 0, u, i, k, a_uk_index, a_k_index
    cdef double a_uk, a_k, d, t, t_prime, delta, t0_t1_d, dEda, dE0
    while w < N:    #
        u = int(data[w, 0])
        i = 0 
        while w < N and data[w, 0] == u:        # over all workouts for the current user
            k = sigma[u][i] 
            a_uk, a_uk_index = get_alpha_ue(theta, u, k, E)
            a_k, a_k_index = get_alpha_e(theta, k, E, U)

            d = data[w, 2]
            t = data[w, 3]
            
            t0_t1_d = (theta_0 + theta_1*d)
            #t_prime = (a_k + a_uk) * (theta_0 + theta_1*d)
            t_prime = (a_k + a_uk) * t0_t1_d

            dEda = 2 * (t_prime - t) * t0_t1_d

            # dE / d_alpha_k
            #dE[a_k_index] += 2 * (t_prime - t) * (theta_0 + theta_1*d);
            dE[a_k_index] += dEda
            
            # dE / d_alpha_uk
            #dE[a_uk_index] += 2 * (t_prime - t) * (theta_0 + theta_1*d);
            dE[a_uk_index] += dEda

            # dE / d_theta_0 and 1
            dE0 = 2 * (t_prime - t) * (a_k + a_uk)
            #dE[-2] += 2 * (t_prime - t) * (a_k + a_uk)
            #dE[-1] += 2 * (t_prime - t) * d * (a_k + a_uk)
            dE[-2] += dE0
            dE[-1] += dE0 * d
            
            w += 1
            i += 1

    # regularization
    cdef double a_k_1, a_uk_1
    for k in range(0, E):
        [a_k, a_k_index] = get_alpha_e(theta, k, E, U)
        if (k < E - 1):
            a_k_1 = get_alpha_e(theta, k + 1, E, U)[0]
            dE[a_k_index] +=  2 * lam * (a_k - a_k_1)
        if (k > 0):
            a_k_1 = get_alpha_e(theta, k - 1, E, U)[0]
            dE[a_k_index] -=  2 * lam * (a_k_1 - a_k)

        for u in range(0, U):
            [a_uk, a_uk_index] = get_alpha_ue(theta, u, k, E)
            if (k < E - 1):
                a_uk_1 = get_alpha_ue(theta, u, k+1, E)[0]
                dE[a_uk_index] +=  2 * lam * (a_uk - a_uk_1)
            if (k > 0):
                a_uk_1 = get_alpha_ue(theta, u, k-1, E)[0]
                dE[a_uk_index] -= 2 * lam * (a_uk_1 - a_uk)

    cdef double t2 = time.time()
    #print "F prime : time taken = ", t2 - t1
    return dE

