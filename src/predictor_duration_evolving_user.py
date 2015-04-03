import gzip
import utils
import numpy as np
import math
import matplotlib.pyplot as plt
from prepare_data_set import prepare_data_set, read_data_as_lists
from plot_data import DataForPlot
import sys
from unit import Unit
import scipy.optimize
import time
import random
import sys
import pyximport; pyximport.install()
from predictor_duration_evolving_user_pyx import F_pyx, Fprime_pyx
import os

def get_user_count(data):
    assert(type(data).__name__ == "list" or type(data).__name__ == "matrix" or type(data).__name__ == "ndarray")
    if (type(data).__name__ == "matrix" or type(data).__name__ == "ndarray"):
        return int(data[-1,0] + 1)   # since user numbers start from 0
    elif (type(data).__name__ == "list"):
        return int(data[-1][0] + 1)
    else:
        raise Exception("invalid type of data..")

def remove_outliers(data, params, param_indices, scale_factors):
    assert(type(data).__name__ == "matrix")
    N1 = data.shape[0]
        
    cols = []; lower_bounds = []; upper_bounds = []

    # remove rows with invalid date (0)
    c = param_indices["date-time"]; cols.append(c); lower_bounds.append(1.0); upper_bounds.append(float("inf"))

    # remove rows distance < 0.06 mi (roughly 100 m)
    c = param_indices["Distance"]; cols.append(c); lower_bounds.append(0.1 / scale_factors[c]); upper_bounds.append(100.0 / scale_factors[c])

    # remove rows with duration < 0.01 hour
    c = param_indices["Duration"]; cols.append(c); lower_bounds.append(0.01 / scale_factors[c]); upper_bounds.append(72.0)

    data = utils.remove_rows_by_condition(data, cols, lower_bounds, upper_bounds)
    
    delete_rows = []
    dist_ind = param_indices["Distance"]
    dur_ind = param_indices["Duration"]
    for d in range(0, data.shape[0]):
        dist = data[d, dist_ind]
        dur = data[d, dur_ind]
        pace = dur * 60.0 / dist;   # min/mi
        #if ((dur > 24.0 or dist > 50.0) and pace <= 500.0):
        #    print "Flagged workout : distance = %f, duration = %f, pace = %f, user = %s" % (round(dist, 4), round(dur, 4), round(pace, 4), data[d, 0])
        if (pace > 200.0):
            delete_rows.append(d)
    print "Deleting %d outlier rows explicitly..", len(delete_rows)
    data = np.delete(data, delete_rows, axis = 0)

    N2 = data.shape[0]
    print "Total %d rows removed during outlier removal.." % (N1 - N2)
    print "Rows remaining..", data.shape[0]
    return data

def is_sorted(data):
    N = len(data)
    for i in range(0, N - 1):
        if (data[i, 1] > data[i+1, 1]):
            return False
    return True

def get_alpha_ue(theta, u, e, E):
    # theta - first UxE elements are per-user per-experience alpha values, next E elements are per experience offset alphas, last 2 are theta0 and theta1
    return theta[u * E + e], u * E + e

def get_alpha_e(theta, e, E, U):
    # theta - first UxE elements are per-user per-experience alpha values, next E elements are per experience offset alphas, last 2 are theta0 and theta1
    return theta[U * E + e], U * E + e

def get_theta_0(theta):
    return theta[-2]

def get_theta_1(theta):
    return theta[-1]

def F(theta, data, lam, E, sigma):
    # error function to be minimized
    # assumes data has 4 columns : user_id, user_number, distance, duration and that it is sorted
    # theta - first UxE elements are per-user per-experience alpha values, next E elements are per experience offset alphas, last 2 are theta0 and theta1
    # sigma - set of experience levels for all workouts for all users.. sigma is a matrix.. sigma(u,i) = e_ui i.e experience level of user u at workout i - these values are NOT optimized by L-BFGS.. they are optimized by DP procedure
    t1 = time.time()
    U = get_user_count(data)
    assert(theta.shape[0] == U * E + E + 2)
    w = 0
    N = data.shape[0]
    f = 0
    theta_0 = get_theta_0(theta)
    theta_1 = get_theta_1(theta)
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
    reg = 0
    for i in range(0, E - 1):
        a_i = get_alpha_e(theta, i, E, U)[0]
        a_i_plus_1 = get_alpha_e(theta, i + 1, E, U)[0]
        diff = a_i - a_i_plus_1
        reg += diff * diff
        for u in range(0, U):
            diff = get_alpha_ue(theta, u, i, E)[0] - get_alpha_ue(theta, u, i+1, E)[0]
            reg += diff * diff
    f += lam * reg
    
    t2 = time.time()
    print "F = %f, time taken = %f" % (f, t2 - t1)
    return f

"""
def Fprime(theta, data, lam, E, sigma):
    # theta - first UxE elements are per-user per-experience alpha values, next E elements are per experience offset alphas, last 2 are theta0 and theta1
    # sigma - set of experience levels for all workouts for all users.. sigma is a matrix.. sigma(u,i) = e_ui i.e experience level of user u at workout i - these values are NOT optimized by L-BFGS.. they are optimized by DP procedure
    t1 = time.time()
    N = data.shape[0]
    U = get_user_count(data)
    assert(theta.shape[0] == U * E + E + 2)
    N = data.shape[0]
    theta_0 = get_theta_0(theta)
    theta_1 = get_theta_1(theta)

    dE = np.array([0.0] * theta.shape[0])

    w = 0
    while w < N:    #
        u = int(data[w, 0])
        i = 0 
        while w < N and data[w, 0] == u:        # over all workouts for the current user
            k = sigma[u][i] 
            a_uk, a_uk_index = get_alpha_ue(theta, u, k, E)
            a_k, a_k_index = get_alpha_e(theta, k, E, U)
            
            d = data[w, 2]
            t = data[w, 3]
            t_prime = (a_k + a_uk) * (theta_0 + theta_1*d)

            # dE / d_alpha_k
            dE[a_k_index] += 2 * (t_prime - t) * (theta_0 + theta_1*d);
            
            # dE / d_alpha_uk
            dE[a_uk_index] += 2 * (t_prime - t) * (theta_0 + theta_1*d);

            # dE / d_theta_0 and 1
            dE[-2] += 2 * (t_prime - t) * (a_k + a_uk)
            dE[-1] += 2 * (t_prime - t) * d * (a_k + a_uk)
            
            w += 1
            i += 1

    # regularization
    for k in range(0, E):
        [a_k, a_k_index] = get_alpha_e(theta, k, E, U)
        delta = 0
        if (k < E - 1):
            a_k_1 = get_alpha_e(theta, k + 1, E, U)[0]
            delta +=  2 * (a_k - a_k_1)
        if (k > 0):
            a_k_1 = get_alpha_e(theta, k - 1, E, U)[0]
            delta -=  2 * (a_k_1 - a_k)
        delta = lam * delta
        dE[a_k_index] += delta;

    t2 = time.time()
    print "F prime : time taken = ", t2 - t1
    return dE
"""

def shuffle_and_split_data_by_user(data, mode, fraction = 0.5):
    # assumes data is numpy matrix form
    # assumes 0th column is the user number
    assert(type(data).__name__ == "matrix")
    assert(mode == "final" or mode == "random")
    i = 0
    N = len(data)
    randomState = np.random.RandomState(seed = 12345)
    n_users = get_user_count(data)
    uins = np.array(range(0, n_users))
    col0 = data[:, 0].A1
    u_indices = list(np.searchsorted(col0, uins))
    u_indices.append(N)
    mask = [0] * N
    for i in range(0, n_users):
        start_u = u_indices[i]
        end_u = u_indices[i+1]
        n_u = end_u - start_u
        assert(n_u > 0)
        if (mode == "random"):
            #perm = randomState.permutation(range(start_u, end_u))
            #end1 = int(math.ceil(fraction * float(n_u)))
            for p in xrange(start_u, end_u): mask[p] = 1
            mask[randomState.randint(low = start_u, high = end_u)] = 2 
        else:   # mode is "final"
            perm = range(start_u, end_u)
            end1 = n_u - 1      # only 1 workout for validation 
            for p in perm[:end1]: mask[p] = 1
            for p in perm[end1:]: mask[p] = 2

    d1_indices = [i for i in range(0, N) if mask[i] == 1]
    d2_indices = [i for i in range(0, N) if mask[i] == 2]
    d1 = data[d1_indices, :]
    d2 = data[d2_indices, :]
    return [d1, d2]

def add_user_number_column(data, param_indices, rare_user_threshold = 1):
    assert(type(data).__name__ == "matrix")
    #data.sort(key=lambda x: x[0])
    #data = utils.sort_matrix_by_col(data, 0)
    # sort first by user id and then date-time
    data = utils.sort_matrix_by_2_cols(data, param_indices["user_id"], param_indices["date-time"])
    n = data.shape[0]
    uin = 0
    i = 0
    uin_col = np.array([0] * n)
    delete_mask = [False] * n
    n_deleted = 0
    while i < n:
        u = data[i, 0]
        start_u = i
        while i < n and data[i, 0] == u:
            #data[i] = [uin] + data[i]
            uin_col[i] = uin
            i += 1
        end_u = i
        n_u = end_u - start_u
        if (n_u > rare_user_threshold):
            # consider only if more than threshold
            uin += 1
        else:
            # mark for deletion
            for j in range(start_u, end_u):
                delete_mask[j] = True
            n_deleted += n_u

    # append col of uins
    uin_col = np.matrix(uin_col).T
    data = np.concatenate((uin_col, data), axis = 1)

    # actually extract only those rows which are not marked for deletion
    delete_indices = [i for i in range(0, n) if delete_mask[i] == True]
    data = np.delete(data, delete_indices, axis = 0)

    # update param_indices
    param_names = [""] * len(param_indices.keys())
    for k, v in param_indices.items():
        param_names[v] = k
    param_indices = string_list_to_dict(["user_number"] + param_names)
    
    print "Number of workouts discarded because very less data for that user : ", n_deleted
    return data, param_indices

def convert_to_numbers(data):
    assert(type(data).__name__ == "list")
    n_invalid_date = 0
    for d in data:
        assert(len(d) == 4)
        d[0] = int(d[0])
        d[1] = float(d[1])
        d[2] = float(d[2])
        try:
            d[3] = utils.parse_date_time(d[3]) 
        except ValueError:
            d[3] = 0        # this row will be removed later in remove_outliers
            n_invalid_date += 1
    print "%d rows had invalid date/time.." % (n_invalid_date)

def make_predictions(data, theta, E, param_indices):
    # use experience levels stored in last column
    N = data.shape[0]
    U = get_user_count(data)
    theta_0 = get_theta_0(theta)
    theta_1 = get_theta_1(theta)
    tpred = np.matrix([0.0] * N).T
    mse = 0.0
    w = 0
    d_ind = param_indices["Distance"]
    t_ind = param_indices["Duration"]
    e_ind = param_indices["experience"]
    while w < N:
        u = int(data[w, 0])
        i = 0
        while w < N and data[w, 0] == u:
            e = data[w, e_ind]
            a_ue = get_alpha_ue(theta, u, e, E)[0]
            a_e = get_alpha_e(theta, e, E, U)[0]
            d = data[w, d_ind]
            tpred[w] = (a_e + a_ue) * (theta_0 + theta_1 * d)
            w += 1
            i += 1
    return tpred

def compute_stats(t_actual, t_pred):
    #assert(t_actual.shape[1] == 1 and t_pred.shape[1] == 1)
    assert(t_actual.shape == t_pred.shape)
    mse = (np.square(t_actual - t_pred)).mean()
    var = np.var(t_actual)
    fvu = mse / var
    r2 = 1 - fvu
    return [mse, var,fvu, r2]

'''
def compute_stats(data, theta, E, sigma):
    N = data.shape[0]
    U = get_user_count(data)
    theta_0 = get_theta_0(theta)
    theta_1 = get_theta_1(theta)
    t = np.array([0.0] * N)
    tpred = np.array([0.0] * N)
    mse = 0.0
    w = 0
    while w < N:
        u = int(data[w, 0])
        i = 0
        while w < N and data[w, 0] == u:
            e = sigma[u][i]
            a_ue = get_alpha_ue(theta, u, e, E)[0]
            a_e = get_alpha_e(theta, e, E, U)[0]
            d = data[w, 2]
            t[w] = data[w, 3]
            tpred[w] = (a_e + a_ue) * (theta_0 + theta_1 * d)
            w += 1
            i += 1
    mse = (np.square(t - tpred)).mean()
    var = np.var(t)
    fvu = mse / var
    r2 = 1 - fvu

    """i
    distances = data[:, 2]
    mat = np.concatenate((np.matrix(distances).T, np.matrix(t).T, np.matrix(tpred).T), axis = 1)
    print mat.shape
    mat = utils.sort_matrix_by_col(mat, 0)
    lim = data.shape[0]
    plt.figure()
    plt.plot(mat[:lim, 0], mat[:lim, 1], label="Actual", marker="o")
    plt.plot(mat[:lim, 0], mat[:lim, 2], label="Predicted", marker="o")
    plt.title("Training")
    plt.legend()
    """

    return [mse, var,fvu, r2]

def compute_stats_validation(data, theta, E, sigma):
    N = data.shape[0]
    U = get_user_count(data)
    theta_0 = get_theta_0(theta)
    theta_1 = get_theta_1(theta)
    t = np.array([0.0] * N)
    tpred = np.array([0.0] * N)
    mse = 0.0
    w = 0
    while w < N:
        u = int(data[w, 0])
        i = 0
        while w < N and data[w, 0] == u:
            e = sigma[u][-1] # consider experience of last workout
            a_ue = get_alpha_ue(theta, u, e, E)[0]
            a_e = get_alpha_e(theta, e, E, U)[0]
            d = data[w, 2]
            t[w] = data[w, 3]
            tpred[w] = (a_e + a_ue) * (theta_0 + theta_1 * d)
            w += 1
            i += 1
    mse = (np.square(t - tpred)).mean()
    var = np.var(t)
    fvu = mse / var
    r2 = 1 - fvu

    """
    distances = data[:, 2]
    mat = np.concatenate((np.matrix(distances).T, np.matrix(t).T, np.matrix(tpred).T), axis = 1)
    print mat.shape
    mat = utils.sort_matrix_by_col(mat, 0)
    lim = data.shape[0]
    plt.figure()
    plt.plot(mat[:lim, 0], mat[:lim, 1], label="Actual", marker="o")
    plt.plot(mat[:lim, 0], mat[:lim, 2], label="Predicted", marker="o")
    plt.title("Validation")
    plt.legend()
    """

    return [mse, var,fvu, r2]
'''

def string_list_to_dict(str_list):
    d = {}
    for p in str_list:
        d[p] = len(d.keys())
    return d

def convert_sec_to_hours(data, param_indices):
    c = param_indices["Duration"]
    data[:, c] = data[:, c] / 3600.0

def normalize(data, cols):
    F = data.shape[1]
    scale_factors = [1.0] * F
    for c in cols:
        scale_factors[c] = np.max(data[:, c])
        data[:, c] /= scale_factors[c]
    return scale_factors

def find_best_path_DP(M):
    E = M.shape[0]
    Nu = M.shape[1]
    #print "Size of M matrix : ", M.shape

    # base case
    D = np.zeros((E, Nu))
    decision = np.zeros((E, Nu))
    for i in range(0, E):
        D[i, 0] = M[i, 0]

    # fill up remaining matrix
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
    leastError = float("inf")
    bestExp = 0
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

def fit_experience_for_all_users(theta, data, E, sigma):
    # sigma - set of experience levels for all workouts for all users.. sigma is a matrix.. sigma(u,i) = e_ui i.e experience level of user u at workout i - these values are NOT optimized by L-BFGS.. they are optimized by DP procedure
    U = get_user_count(data)
    N = data.shape[0]
    row = 0
    theta_0 = get_theta_0(theta)
    theta_1 = get_theta_1(theta)
    changed = False
    for u in range(0, U):
        Nu = 0
        row_u = row
        while (row < N and data[row, 0] == u):
            Nu += 1
            row += 1
        #print "Number of workouts for this user : ", Nu

        # populate M
        M = np.zeros((E, Nu))
        for j in range(0, Nu):  # over all workouts for this user
            t = data[row_u + j, 3]    # actual time for that workout
            d = data[row_u + j, 2]
            for i in range(0, E):       # over all experience levels
                a_ue = get_alpha_ue(theta, u, i, E)[0]
                a_e = get_alpha_e(theta, i, E, U)[0]
                tprime = (a_e + a_ue) * (theta_0 + theta_1 * d)
                diff = t - tprime
                M[i, j] = diff * diff


        [minError, bestPath] = find_best_path_DP(M)
        #print minError, bestPath
        # update sigma matrix using bestPath
        for i in range(0, Nu):
            if (sigma[u][i] != bestPath[i]):
                sigma[u][i] = bestPath[i]
                changed = True
                #print "Updated sigma[%d, %d].." % (u, i)
                #print sigma[u, :]
        
    return changed

def experience_check(theta, data, E):
    # intuitively, given the same distance, the predicted duration must be lower with increasing experience. This functions tries to test this.
    d = 12.32
    U = get_user_count(data)
    theta0 = get_theta_0(theta)
    theta1 = get_theta_0(theta)
    for u in range(0, U):
        predictions = [0.0] * E
        for i in range(0, E):
            a_e = get_alpha_e(theta, i, E, U)[0]
            a_ue = get_alpha_ue(theta, u, i, E)[0]
            predictions[i] = (a_e + a_ue) * (theta0 + theta1 * d)
        # check if predictions is monotonically decreasing
        for i in range(0, E-1):
            assert(predictions[i] > predictions[i+1])

def learn_cpp(data, lam1, lam2):
    # write data to file
    data_file = "data.txt"
    np.savetxt(data_file, data)

    # call cpp executable
    infile = data_file
    outfile = "model.txt"
    exec_name = "./predictor_duration_evolving_user_cpp"
    command = "%s %s %s %s %s" % (exec_name, infile, str(lam1), str(lam2), outfile)
    print "Running command %s" % (command)
    assert(os.system(command) == 0)
    print "Done with learning.."

    # read output from file
    print "Reading learned model from file.."
    with open(outfile) as f:
        for line in f:
            parts = line.strip().split("=")
            assert(len(parts) == 2)
            k, v = parts
            if (k == "theta"):
                theta = eval(v)
            elif (k == "sigma"):
                sigma = eval(v)
            elif (k == "E"):
                E = eval(v)
            else:
                raise Exception("error in parsing outfile of cpp executable")
    return theta, sigma, E

def learn(data, lam1, lam2):
    E = 3
    #lam1 = 1.0
    #lam2 = 0.1
    check_grad = False
    F_fn = F_pyx
    Fprime_fn = Fprime_pyx

    print "@E = %d,lam1 = %f,lam2 = %f" % (E, lam1, lam2)
    U = get_user_count(data)
    randomState = np.random.RandomState(12345)
    #theta = np.array([1.0] * (U * E + E + 2))
    theta = randomState.rand(U * E + E + 2)
    workouts_per_user = get_workouts_per_user(data)
    sigma = []
    for u in range(0, U):
        sigma.append(list(np.sort(randomState.randint(low = 0, high = E, size = (workouts_per_user[u])))))
        #sigma.append([0.0] * workouts_per_user[u])
    #sigma = np.array(sigma)

    # check grad first
    if (check_grad == True):
        print "Checking gradient.."
        our_grad = np.linalg.norm(Fprime_fn(theta, data, lam1, lam2, E, sigma), ord = 2)
        numerical = np.linalg.norm(scipy.optimize.approx_fprime(theta, F_fn, np.sqrt(np.finfo(np.float).eps), data, lam1, lam2, E, sigma), ord = 2)
        ratio = our_grad / numerical
        print "Ratio = ", ratio
        assert(abs(1.0 - ratio) < 1e-4)
        sys.exit(0)

    n_iter = 0

    changed = True
    while changed and n_iter < 100:
        print "Iteration %d.." % (n_iter)

        # 1. optimize theta
        [theta, E_min, info] = scipy.optimize.fmin_l_bfgs_b(F_fn, theta, Fprime_fn, args = (data, lam1, lam2, E, sigma),  maxfun=100, maxiter=100, iprint=1, disp=0)

        # 2. use DP to fit experience levels
        changed = fit_experience_for_all_users(theta, data, E, sigma)

        print "@E = ", E_min
        
        n_iter += 1

    print "norm of final theta = ", np.linalg.norm(theta, ord = 2)
    print "final value of error function = ", F_pyx(theta, data, lam1, lam2, E, sigma)
    print "final value of norm of gradient function = ", np.linalg.norm(Fprime_pyx(theta, data, lam1, lam2, E, sigma), ord = 2)

    return theta, sigma, E

def get_workouts_per_user(data):
    assert(type(data).__name__ == "matrix" or type(data).__name__ == "ndarray")
    N = data.shape[0]
    U = get_user_count(data)
    uins = np.array(range(0, U))
    col0 = data[:, 0]
    u_indices = list(np.searchsorted(col0, uins))
    u_indices.append(N)
    workouts_per_user = [0] * U
    for i in range(0, U):
        start_u = u_indices[i]
        end_u = u_indices[i+1]
        workouts_per_user[i] = end_u - start_u
    return workouts_per_user

def add_experience_column_to_train_set(train_set, sigma, param_indices):
    np.set_printoptions(suppress = True)
    np.set_printoptions(precision = 2)
    U = get_user_count(train_set)
    #assert(U == len(sigma))
    N = train_set.shape[0]
    exp_col = np.matrix([0] * N).T
    w = 0
    for u in range(0, U):
        Nu = len(sigma[u])
        for i in range(0, Nu):
            exp_col[w] = sigma[u][i]
            w += 1
    train_set = np.concatenate((train_set, exp_col), axis = 1)
    if (not param_indices.has_key("experience")):
        param_indices["experience"] = train_set.shape[1] - 1    # last column
    else:
        assert(param_indices["experience"] == train_set.shape[1] - 1)
    return train_set

def add_experience_column_to_test_set(test_set, train_set, param_indices, mode = "final"):
    # test_set can be validation set or test set 
    # if mode is final, there must be only 1 workout per user in test_set. this will be assigned the experience of the last workout in the training set for that user
    # if mode is random, every workout will be assigned the experience of the closest workout (closest in time) of that user
    print "Mode : ", mode
    N_train = train_set.shape[0]
    N_test = test_set.shape[0]
    U = get_user_count(train_set)
    exp_col = np.matrix([0] * test_set.shape[0]).T
    exp_ind = param_indices["experience"]
    datetime_ind = param_indices["date-time"]
    uins = np.array(range(0, U))
    train_u_indices = list(np.searchsorted(np.matrix(train_set[:, 0]).A1, uins)); train_u_indices.append(N_train)
    test_u_indices = list(np.searchsorted(np.matrix(test_set[:, 0]).A1, uins)); test_u_indices.append(N_test)
    if (mode == "final"):
        assert (U == get_user_count(test_set) and test_set.shape[0] == U)
        for u in range(0, U):
            #Nu = train_u_indices[u+1] - train_u_indices[u]
            assert(test_set[u][0] == u)     # since we have exactly 1 workout per user in final mode
            exp_col[u] = train_set[train_u_indices[u+1] - 1, exp_ind]
    elif (mode == "random"):
        w = 0
        N_test = test_set.shape[0]
        for u in range(0, U):
            train_start_u = train_u_indices[u];  train_end_u = train_u_indices[u+1]
            train_Nu = train_end_u - train_start_u
            train_date_times = train_set[train_start_u:train_end_u, datetime_ind].A1

            test_start_u = test_u_indices[u];  test_end_u = test_u_indices[u+1]
            test_Nu = test_end_u - test_start_u
            assert(test_Nu == 1)
            for i in range(0, test_Nu):
                curr_dt = test_set[test_start_u + i, datetime_ind]
                j = np.searchsorted(train_date_times, curr_dt)
                if j == train_Nu:
                    j = j - 1   # closest will be last workout
                elif j > 0:
                    if (abs(train_set[train_start_u + j - 1, datetime_ind] - curr_dt) 
                            < abs(train_set[train_start_u + j, datetime_ind] - curr_dt)):
                        j = j - 1
                else:   # if j == 0
                    pass        # closest will be 0th workout
                e = train_set[train_start_u + j, exp_ind]
                exp_col[w] = e
                w += 1
    else:
        raise Exception("Invalid mode of testing..")

    test_set = np.concatenate((test_set, exp_col), axis = 1)
    if (not param_indices.has_key("experience")):
        param_indices["experience"] = train_set.shape[1] - 1    # last column
    else:
        assert(param_indices["experience"] == train_set.shape[1] - 1)
    return test_set

def prepare(infile, outfile, mode):
    sport = "Running"
    params = ["user_id","Distance", "Duration", "date-time"]
    param_indices = string_list_to_dict(params)
    
    print "Reading data.."
    data = read_data_as_lists(infile, sport, params)
    convert_to_numbers(data)   # convert from strings to numbers
    
    print "Converting data matrix to numpy format"
    data = np.matrix(data)
    convert_sec_to_hours(data, param_indices)
    cols = []
    #cols.append(param_indices["Duration"])
    #cols.append(param_indices["Distance"])
    scale_factors = normalize(data, cols)
    print "Scale factors : ", scale_factors

    print "Removing outliers.."
    data = remove_outliers(data, params, param_indices, scale_factors)
    
    print "Adding user numbers.."
    data, param_indices = add_user_number_column(data, param_indices, rare_user_threshold = 10)    # add a user number
    assert(param_indices == string_list_to_dict(["user_number"] + params))
        
    print "Splitting data into training, validation and test"
    #[train_set, val_set] = shuffle_and_split_data_by_user(data, mode)
    print "Full data set contains %d workouts.." % (data.shape[0])
    [d1, test_set] = shuffle_and_split_data_by_user(data, mode, fraction = 0.7)
    [train_set, val_set] = shuffle_and_split_data_by_user(d1, mode, fraction = 0.57)    # train = 40 % of total, val = 30 % of total, test = 30 % of total.. For "final" mode, these values are ignored
    print "Train set contains %d workouts\nValidation contains %d workouts\nTest set contains %d workouts" % (train_set.shape[0], val_set.shape[0], test_set.shape[0])
    assert(not np.array_equal(train_set, test_set))
    assert(not np.array_equal(val_set, test_set))
    assert(get_user_count(train_set) == get_user_count(val_set) and get_user_count(val_set) == get_user_count(test_set))
    
    print "Saving data to disk"
    np.savez(outfile, train_set = train_set, val_set = val_set, param_indices = param_indices)

def plot_data(data, predictions, param_indices, title = ""):
    #dur_ind = param_indices["Duration"]
    #x = list(data[:, dur_ind])
    #plt.hist(x, bins = 200, range = [0, 24])

    # plot distance on X axis, predicted and actual duration on Y axis
    t_pred = predictions
    t_actual = data[:, param_indices["Duration"]]
    assert(t_actual.shape == t_pred.shape)
    distances = data[:, param_indices["Distance"]]
    mat = np.concatenate((distances, t_actual, t_pred), axis = 1)
    mat = utils.sort_matrix_by_col(mat, 0)
    lim = data.shape[0]
    plt.plot(mat[:lim, 0], mat[:lim, 1], label="Actual", marker="o")
    plt.plot(mat[:lim, 0], mat[:lim, 2], label="Predicted", marker="o")
    plt.title(title)
    plt.legend()

def check_sorted(data, param_indices):
    N = data.shape[0]
    ind_u = param_indices["user_number"]
    ind_t = param_indices["date-time"]
    i = 0
    while (i < N):
        u = (data[i, ind_u])
        i += 1
        while i < N and data[i, ind_u] == u:
            assert(data[i, ind_t] >= data[i - 1, ind_t])
            i += 1
        if (i < N):
            assert(data[i - 1, ind_u] < data[i, ind_u])

if __name__ == "__main__":
    t1 = time.time()
    # prepare data set.. Run once and comment it out if running multiple times with same settings
    #infile = "temp.gz"
    #infile = "../../data/all_workouts_train_and_val_condensed.gz"
    #infile = "synth_evolving_user_model.gz"
    
    #infile = "endoMondo5000_workouts_condensed.gz"
    infile = "../../data/all_workouts_condensed.gz"
    mode = "random"  # can be "final" or "random"
    outfile = infile + mode + ".npz"

    #prepare(infile, outfile, mode)

    print "Loading data from file.."
    data = np.load(outfile)
    train_set = data["train_set"]
    val_set = data["val_set"]
    param_indices = data["param_indices"][()]
    print "Doing sorted check on train and val sets.."
    check_sorted(train_set, param_indices)
    check_sorted(val_set, param_indices)

    print "Number of users = ", get_user_count(train_set)
    print "Training set has %d examples" % (train_set.shape[0])
    print "Validation set has %d examples" % (val_set.shape[0])

    print "Training.."
    lam1 = float(sys.argv[1])
    lam2 = float(sys.argv[2])
    #theta, sigma, E = learn(train_set, lam1, lam2)
    theta, sigma, E = learn_cpp(train_set, lam1, lam2)
    np.savez("model.npz", theta = theta, sigma = sigma, E = E)
    
    print "Loading model.."
    model = np.load("model.npz")
    theta = model["theta"]
    sigma = model["sigma"]
    E = model["E"]

    # add the experience level to each workout in train, validation and test set
    print "Adding experience levels to data matrices"
    train_set = add_experience_column_to_train_set(train_set, sigma, param_indices)
    val_set = add_experience_column_to_test_set(val_set, train_set, param_indices, mode = mode)

    print "Making predictions.."
    train_pred = make_predictions(train_set, theta, E, param_indices)
    val_pred = make_predictions(val_set, theta, E, param_indices)
    print param_indices

    print "Computing statistics"
    [mse, var, fvu, r2] = compute_stats(train_set[:, param_indices["Duration"]], train_pred)
    print "\n@Training Examples = %d,MSE = %f,Variance = %f,FVU = %f,R2 = 1 - FVU = %f\n" % (train_set.shape[0],mse, var, fvu, r2)
    [mse, var, fvu, r2] = compute_stats(val_set[:, param_indices["Duration"]], val_pred)
    print "@Validation Examples = %d,MSE = %f,Variance = %f,FVU = %f,R2 = 1 - FVU = %f\n" % (val_set.shape[0],mse, var, fvu, r2)

    t2 = time.time()
    print "@Total time taken = ", t2 - t1

    plt.figure()
    plt.subplot(1,2,0)
    plot_data(train_set, train_pred, param_indices, title = "Training set")
    plt.subplot(1,2,1)
    plot_data(val_set, val_pred, param_indices, title = "Validation set")

    #plt.show()
