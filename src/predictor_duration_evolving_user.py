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
#import pyximport; pyximport.install()
#from predictor_duration_evolving_user_pyx import Fprime_pyx, F_pyx

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
    #assert(missing_data_mode == "ignore" or missing_data_mode == "substitute")
    #if (missing_data_mode == "ignore"):
    #    assert(len(params) == Xy.shape[1])
    #else:
    #    assert(2.0 * len(params) - 1 == Xy.shape[1])
    
    cols = []; lower_bounds = []; upper_bounds = []

    # remove rows distance < 0.01 mi
    c = param_indices["Distance"]; cols.append(c); lower_bounds.append(0.01 / scale_factors[c]); upper_bounds.append(float("inf"))

    # remove rows with duration < 0.01 hours
    c = param_indices["Duration"]; cols.append(c); lower_bounds.append(0.01 / scale_factors[c]); upper_bounds.append(float("inf"))

    # remove rows with other parameters < 0.1
    #c = param_indices["pace(avg)"]; cols.append(c); lower_bounds.append(0.1); upper_bounds.append(float("inf"))
    #c = param_indices["Total Ascent"]; cols.append(c); lower_bounds.append(float("-inf")); upper_bounds.append(15000)
    #c = param_indices["Total Descent"]; cols.append(c); lower_bounds.append(float("-inf")); upper_bounds.append(15000)
    #c = param_indices["alt(avg)"]; cols.append(c); lower_bounds.append(-1000); upper_bounds.append(30000)

    data = utils.remove_rows_by_condition(data, cols, lower_bounds, upper_bounds)
    N2 = data.shape[0]
    print "%d rows removed during outlier removal.." % (N1 - N2)
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
            e = sigma[u, i] 
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
    #print "E = %f, time taken = %f" % (e, t2 - t1)
    return f

def Fprime_slow(theta, data, lam, E, sigma):
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
            k = sigma[u, i] 
            a_uk, a_uk_index = get_alpha_ue(theta, u, k, E)
            a_k, a_k_index = get_alpha_e(theta, k, E, U)
            
            d = data[w, 2]
            t = data[w, 3]
            t_prime = (a_k + a_uk) * (theta_0 + theta_1*d)

            # dE / d_alpha_k
            dE[a_k_index] += 2 * (t_prime - t) + a_k*theta_0 + a_k*theta_1*d;
            
            # dE / d_alpha_uk
            dE[a_uk_index] += 2 * (t_prime - t) + a_uk*theta_0 + a_uk*theta_1*d;

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
    #print "E prime : time taken = ", t2 - t1
    return dE

def shuffle_and_split_data_by_user(data):
    # assumes data is numpy matrix form
    # assumes 0th column is the user number
    assert(type(data).__name__ == "matrix")
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
        perm = randomState.permutation(range(start_u, end_u))
        #end1 = int(math.ceil((fraction * float(n_u))))
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

    #print "Number of users = ", get_user_count(data)
    print "Number of workouts discarded because very less data for that user : ", n_deleted
    return data

def convert_to_numbers(data):
    assert(type(data).__name__ == "list")
    for d in data:
        assert(len(d) == 4)
        d[0] = int(d[0])
        d[1] = float(d[1])
        d[2] = float(d[2])
        d[3] = utils.parse_date_time(d[3]) 

def compute_stats(data, theta):
    N = data.shape[0]
    theta_0 = theta[-2]
    theta_1 = theta[-1]
    t = np.array([0.0] * N)
    tprime = np.array([0.0] * N)
    mse = 0.0
    for i in range(0, N):
        u = data[i, 0]
        alpha = theta[u]
        d = data[i, 2]
        t[i] = data[i, 3]
        tprime[i] = alpha * (theta_0 + theta_1 * d)
    mse = (np.square(t - tprime)).mean()
    var = np.var(t)
    fvu = mse / var
    r2 = 1 - fvu
    return [mse, var,fvu, r2]

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
            Nu += 1; row += 1;

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
                    o1 = M[m-1, n] + D[m-1, n-1]
                o2 = M[m, n] + D[m, n-1]
                if (o1 < o2):
                    D[m, n] = o1
                    decision[m, n] = m - 1
                else:
                    D[m, n] = o2
                    decision[m, n] = m

        # trace path
        leastError = float("inf")
        bestExp = 0
        # first compute for last workout
        for i in range(0, E):
            if (D[i, Nu-1] < leastError):
                leastError = D[i, Nu-1]
                bestExp = i
        if (sigma[u, Nu - 1] != bestExp):
            sigma[u, Nu - 1] = bestExp
            changed = True
        # now trace for remaining workouts backwards
        for i in range(Nu - 2, -1, -1):
            bestExp = decision[sigma[u, i+1], i+1]
            if (sigma[u, i] != bestExp):
                sigma[u, i] = bestExp
                changed = True

    return changed

def learn(data):
    U = get_user_count(data)
    E = 5
    theta = np.array([1.0] * (U * E + E + 2))
    sigma = np.zeros((U, 400))
    changed = True
    lam = 0.0

    # check grad first
    print "Checking gradient.."
    error = scipy.optimize.check_grad(F, Fprime_slow, theta, data, lam, E, sigma)
    print "Error = ", error
    print "Gradient = ", np.linalg.norm(Fprime_slow(theta, data, lam, E, sigma), ord = 2)
    assert(error < 0.0001)

    changed = False
    while changed:
        # 1. optimize theta
        [theta, E_min, info] = scipy.optimize.fmin_l_bfgs_b(F, theta, Fprime_slow, args = (data, lam, E, sigma),  maxfun=100, maxiter=100, iprint=1, disp=1, factr=10)
        print info

        # 2. use DP to fit experience levels
        changed = fit_experience_for_all_users(theta, data, E, sigma)

    return theta

def prepare(infile, outfile):
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
    data = add_user_number_column(data, param_indices, rare_user_threshold = 3)    # add a user number
        
    print "Splitting data into training and validation"
    [d1, d2] = shuffle_and_split_data_by_user(data)
    
    print "Saving data to disk"
    np.savez(outfile, d1 = d1, d2 = d2)


if __name__ == "__main__":
    t1 = time.time()
    # prepare data set.. Run once and comment it out if running multiple times with same settings
    infile = "endoMondo5000_workouts_condensed.gz"
    #infile = "../../data/all_workouts_train_and_val_condensed.gz"
    #infile = "synth1.gz"
    outfile = infile + ".npz"
    #e_fn = E_pyx
    #eprime_fn = Eprime_pyx

    prepare(infile, outfile)

    print "Loading data from file.."
    data = np.load(outfile)
    train = data["d1"]
    val = data["d2"]
    n_users = get_user_count(train)
    assert(get_user_count(train) == get_user_count(val))
    theta = [1.0] * (n_users + 2)
    lam = 0.0    # regularization

    #np.set_printoptions(precision=3)
    #np.set_printoptions(suppress=True)
    #print train[:10, :]

    print "Training.."

    theta = learn(train)

    print "Computing predictions and statistics"
    [mse, var, fvu, r2] = compute_stats(train, theta)
    print "\nStats for training data : \n# Examples = %d\nMSE = %f\nVariance = %f\nFVU = %f\nR2 = 1 - FVU = %f\n" % (train.shape[0],mse, var, fvu, r2)
    [mse, var, fvu, r2] = compute_stats(val, theta)
    print "\nStats for val data : \n# Examples = %d\nMSE = %f\nVariance = %f\nFVU = %f\nR2 = 1 - FVU = %f\n" % (val.shape[0],mse, var, fvu, r2)

    t2 = time.time()
    print "Total time taken = ", t2 - t1
    sys.exit(0)

    # plots for regularization
    """
    all_lam = []
    all_r2_train = []
    all_r2_val = []
    while lam > 1e-7:
        all_lam.append(lam)
        theta = [1.0] * (n_users + 2)
        [theta, E_min, info] = scipy.optimize.fmin_l_bfgs_b(e_fn, theta, eprime_fn, args = (train, lam), maxfun=100)
        [mse, var, fvu, r2] = compute_stats(train, theta)
        all_r2_train.append(r2)
        print "\nStats for training data : \n# Examples = %d\nMSE = %f\nVariance = %f\nFVU = %f\nR2 = 1 - FVU = %f\n" % (train.shape[0],mse, var, fvu, r2)
        [mse, var, fvu, r2] = compute_stats(val, theta)
        all_r2_val.append(r2)
        lam = lam / 2.0
        print "\nStats for val data : \n# Examples = %d\nMSE = %f\nVariance = %f\nFVU = %f\nR2 = 1 - FVU = %f\n" % (val.shape[0],mse, var, fvu, r2)

    print "all_lam =", all_lam
    print "all_r2_train = ", all_r2_train
    print "all_r2_val = ", all_r2_val

    plt.figure()
    plt.plot(all_lam, all_r2_train, label="Training R2")
    plt.plot(all_lam, all_r2_val, label="Validation R2")
    plt.legend()
    plt.show() 
    """
