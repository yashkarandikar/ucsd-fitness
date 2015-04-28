import numpy as np
cimport numpy as np
import time as time
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


def find_best_path_DP(np.ndarray[DTYPE_t, ndim=2] M):
    cdef int E = M.shape[0]
    cdef int Nu = M.shape[1]
    cdef int i
    #print "Size of M matrix : ", M.shape

    # base case
    cdef np.ndarray[DTYPE_t, ndim=2] D 
    D = np.zeros((E, Nu))
    cdef np.ndarray[DTYPE_t, ndim=2] decision
    decision = np.zeros((E, Nu))
    for i in range(0, E):
        D[i, 0] = M[i, 0]

    # fill up remaining matrix
    cdef int n, m
    cdef double o1, o2
    for n in range(1, Nu):
        for m in range(0, E):
            o1 = float("inf")
            if (m > 0):
                o1 = D[m-1, n-1]
            o2 = D[m, n-1]
            if (o1 < o2):
                D[m, n] = M[m, n] + o1
                decision[m, n] = m - 1
            else:
                D[m, n] = M[m, n] + o2
                decision[m, n] = m

    # trace path
    cdef double leastError = float("inf")
    cdef int bestExp = 0
    # first compute for last workout
    for i in range(0, E):
        if (D[i, Nu-1] < leastError):
            leastError = D[i, Nu-1]
            bestExp = i
    path = [bestExp]
    # now trace for remaining workouts backwards
    for i in range(Nu - 2, -1, -1):
        bestExp = decision[path[0], i+1]
        path = [bestExp] + path

    # check that path is monotonically increasing
    for i in range(0, len(path) - 1):
        assert(path[i] <= path[i+1])

    return [leastError, path]

def fit_tiredness_for_all_workouts_pyx(np.ndarray[DTYPE_t, ndim=1] theta, np.ndarray[DTYPE_t, ndim=2] data, np.int_t E, sigma, np.ndarray[DTYPE_t, ndim=2] hr = None, np.ndarray[long, ndim=1] last_e = None, use_features = True):
    from predictor_many_insthr_evolving import get_theta_0, get_theta_1, get_alpha_e, get_alpha_ue, get_workout_count
    # sigma - set of experience levels for all workouts for all users.. sigma is a matrix.. sigma(u,i) = e_ui i.e experience level of user u at workout i - these values are NOT optimized by L-BFGS.. they are optimized by DP procedure
    cdef int U = get_workout_count(data)
    cdef int N = data.shape[0]
    cdef int row = 0, u, Nu, row_u, j, n_E, low_E
    cdef double theta_0 = get_theta_0(theta)
    cdef double theta_1 = get_theta_1(theta)
    cdef double d, a_ue, a_e, tprime, diff, minError
    cdef np.ndarray[DTYPE_t, ndim=2] M
    cdef int n_skipped = 0
    changed = False
    for u in xrange(0, U):
        Nu = 0
        row_u = row
        while (row < N and data[row, 0] == u):
            Nu += 1
            row += 1
        #print "Number of workouts for this user : ", Nu

        low_E = 0
        if (last_e is not None):
            low_E = int(last_e[u])
        n_E = E - low_E

        if (n_E > 1):
            # populate M
            M = np.zeros((n_E, Nu))
            for j in xrange(0, Nu):  # over all workouts for this user
                if (hr is None):
                    t = data[row_u + j, 3]    # actual time for that workout
                else:
                    t = hr[row_u + j, 0]
                d = 0.0
                if (use_features):
                    d = data[row_u + j, 2]
                for i in xrange(low_E, E):       # over all experience levels
                    a_ue = get_alpha_ue(theta, u, i, E)[0]
                    a_e = get_alpha_e(theta, i, E, U)[0]
                    tprime = (a_e + a_ue) * (theta_0 + theta_1 * d)
                    diff = t - tprime
                    M[i - low_E, j] = diff * diff


            [minError, bestPath] = find_best_path_DP(M)
            bestPath = [e + low_E for e in bestPath]
            #print minError, bestPath
            # update sigma matrix using bestPath
            for i in xrange(0, Nu):
                if (sigma[u][i] != bestPath[i]):
                    sigma[u][i] = bestPath[i]
                    changed = True
                    #print "Updated sigma[%d, %d].." % (u, i)
                    #print sigma[u, :]
        else:
            n_skipped += 1
        if (u % 10000 == 0):
            print "Done %d out of %d, (%d skipped)" % (u, U, n_skipped)
        
    print "Done fitting tiredness levels.. %d workouts skipped " % (n_skipped)
    return changed


def make_predictions_separate_sigma_pyx(np.ndarray[DTYPE_t, ndim=2] data, np.ndarray[DTYPE_t, ndim=1] theta, np.int_t E, param_indices, sigma, use_features):
    from predictor_many_insthr_evolving import get_theta_0, get_theta_1, get_alpha_e, get_alpha_ue, get_workout_count
    # use experience levels stored separately in sigma
    cdef int N = data.shape[0]
    cdef int U = get_workout_count(data)
    cdef double theta_0 = get_theta_0(theta)
    cdef double theta_1 = get_theta_1(theta)
    cdef np.ndarray[DTYPE_t, ndim=2] tpred = np.matrix([0.0] * N).T
    cdef double mse = 0.0, e, a_ue, a_e, d
    cdef int w = 0, u, i
    cdef int d_ind = param_indices["distance"]
    cdef int t_ind = param_indices["hr"]
    while w < N:
        u = int(data[w, 0])
        i = 0
        while w < N and data[w, 0] == u:
            #e = data[w, e_ind]
            e = sigma[u][i]
            a_ue = get_alpha_ue(theta, u, e, E)[0]
            a_e = get_alpha_e(theta, e, E, U)[0]
            d = 0.0
            if (use_features):
                d = data[w, d_ind]
            tpred[w] = (a_e + a_ue) * (theta_0 + theta_1 * d)
            w += 1
            i += 1
            if (w % 1000000 == 0):
                print "%d data points done.." % (w)
    return tpred

def make_predictions_pyx(np.ndarray[DTYPE_t, ndim=2] data, np.ndarray[DTYPE_t, ndim=1] theta, np.int_t E, param_indices, use_features):
    # use experience levels stored in last column
    from predictor_insthr_evolving import get_theta_0, get_theta_1, get_alpha_e, get_alpha_ue, get_workout_count
    cdef int N = data.shape[0]
    cdef int U = get_workout_count(data)
    cdef double theta_0 = get_theta_0(theta)
    cdef double theta_1 = get_theta_1(theta)
    cdef np.ndarray[DTYPE_t, ndim=2] tpred = np.matrix([0.0] * N).T
    cdef double mse = 0.0, a_ue, a_e, d
    cdef int w = 0
    cdef int d_ind = param_indices["distance"]
    cdef int t_ind = param_indices["hr"]
    cdef int e_ind = param_indices["experience"]
    cdef int u, i, e
    
    for w in xrange(0, N):
        u = int(data[w, 0])
        i = 0
        e = int(data[w, e_ind])
        a_ue = get_alpha_ue(theta, u, e, E)[0]
        a_e = get_alpha_e(theta, e, E, U)[0]
        if (use_features):
            d = data[w, d_ind]
        tpred[w, 0] = (a_e + a_ue) * (theta_0 + theta_1 * d)
        w += 1
        i += 1
        if (w % 1000000 == 0):
            print "%d data points done.." % (w)

    return tpred
