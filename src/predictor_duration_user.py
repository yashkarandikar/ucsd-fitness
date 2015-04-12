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
from predictor_duration_user_pyx import Eprime_pyx, E_pyx
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

"""
def E(theta, data, lam):
    # error function to be minimized
    # assumes data has 4 columns : user_id, user_number, distance, duration and that it is sorted
    t1 = time.time()
    i = 0
    N = data.shape[0]
    e = 0
    theta_0 = theta[-2]
    theta_1 = theta[-1]
    while i < N:
        u = int(data[i, 0])
        alpha = theta[u]
        while i < N and data[i, 0] == u:
            d = data[i, 2]
            t = data[i, 3]
            e += math.pow(alpha * (theta_0 + theta_1 * d) - t, 2)
            i += 1
    # add regularization norm
    e += lam * theta.dot(theta)
    
    t2 = time.time()
    print "E = %f, time taken = %f" % (e, t2 - t1)
    return e

def Eprime(theta, data, lam):
    t1 = time.time()
    N = data.shape[0]
    #n_users = int(data[-1, 0]) + 1
    n_users = get_user_count(data)
    assert(theta.shape[0] == n_users + 2)
    theta_0 = theta[-2]
    theta_1 = theta[-1]
    #dE = np.array([0.0] * len(theta))
    dE = [0.0] * len(theta)
    i = 0
    col0 = np.ravel(data[:, 0])
    uins = np.array(range(0, n_users))
    u_indices = list(np.searchsorted(col0, uins))
    u_indices.append(N)
    
    dE_theta0 = 0.0
    dE_theta1 = 0.0

    for i in xrange(0, n_users):
        start_u = u_indices[i]
        end_u = u_indices[i+1]
        alpha_u = theta[i]
        for j in xrange(start_u, end_u):
            t = data[j, -1]
            d = data[j, 2]
            
            t0_t1_d = theta_0 + theta_1 * d
            a_t0_t1_d = alpha_u * t0_t1_d
            dE[i] = dE[i] + 2 * (a_t0_t1_d - t) * t0_t1_d

            # dE / d_theta_0 and 1
            dE0 = 2 * alpha_u * (a_t0_t1_d - t)
            dE_theta0 += dE0
            dE_theta1 += dE0 * d
        
    dE[-2] = dE_theta0
    dE[-1] = dE_theta1

    # regularization
    dE = dE + lam * np.multiply(dE, (2 * theta))

    t2 = time.time()
    print "E prime : time taken = ", t2 - t1
    return np.array(dE)


def Eprime_slow(theta, data):
    t1 = time.time()
    N = data.shape[0]
    n_users = get_user_count(data)
    assert(len(theta) == n_users + 2)
    theta_0 = theta[-2]
    theta_1 = theta[-1]
    dE = np.array([0.0] * len(theta))
    i = 0
    while i < N:
        u = int(data[i, 0])
        alpha_u = theta[u]
        while i < N and data[i, 0] == u:
            # dE / d_alpha_u
            d = data[i, 2]
            t = data[i, 3]
            dE[u] += 2 * (alpha_u * (theta_0 + theta_1 * d) - t) * (theta_0 + theta_1 * d)

            # dE / d_theta_0 and 1
            dE[-2] += 2 * alpha_u * (alpha_u*(theta_0 + theta_1 * d) - t)
            dE[-1] += 2 * alpha_u * d * (alpha_u*(theta_0 + theta_1 * d) - t)
            
            i += 1
    t2 = time.time()
    #print "E prime : time taken = ", t2 - t1
    return dE
"""

def shuffle_and_split_data_by_user(data, fraction = 0.5):
    # assumes data is numpy matrix form
    # assumes 0th column is the user number
    assert(type(data).__name__ == "matrix")
    i = 0
    N = len(data)
    randomState = np.random.RandomState(seed = 12345)
    #n_users = int(data[-1, 0]) + 1
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
        #perm = range(start_u, end_u)
        #random.shuffle(perm)
        perm = randomState.permutation(range(start_u, end_u))
        end1 = int(math.ceil((fraction * float(n_u))))
        for p in perm[:end1]: mask[p] = 1
        for p in perm[end1:]: mask[p] = 2
        #d1_indices = d1_indices + perm[:end1]
        #d2_indices = d2_indices + perm[end1:]
        #if (i % 10000 == 0):
            #print "Done with %d users " % (i)

    d1_indices = [i for i in range(0, N) if mask[i] == 1]
    d2_indices = [i for i in range(0, N) if mask[i] == 2]
    d1 = data[d1_indices, :]
    d2 = data[d2_indices, :]
    return [d1, d2]

def add_user_number_column(data, rare_user_threshold = 1):
    assert(type(data).__name__ == "matrix")
    #data.sort(key=lambda x: x[0])
    data = utils.sort_matrix_by_col(data, 0)
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
        assert(len(d) == 3)
        d[0] = int(d[0])
        d[1] = float(d[1])
        d[2] = float(d[2])

def compute_stats(data, theta):
    U = get_user_count(data)
    assert(theta.shape[0] == 3 + U)
    N = data.shape[0]
    alpha_all = theta[-3]
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
        tprime[i] = (alpha + alpha_all) * (theta_0 + theta_1 * d)
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
    
def prepare(infile, outfile):
    sport = "Running"
    params = ["user_id","Distance", "Duration"]
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
    data = add_user_number_column(data, rare_user_threshold = 3)    # add a user number
        
    print "Splitting data into training and validation"
    [d1, d2] = shuffle_and_split_data_by_user(data)
    
    print "Saving data to disk"
    np.savez(outfile, train_set = d1, val_set = d2)

def learn_cpp(data, lam):
    # write data to file
    data_file = "data.txt"
    np.savetxt(data_file, data)

    # call cpp executable
    infile = data_file
    outfile = "model.txt"
    exec_name = "./predictor_duration_user_cpp"
    command = "%s %s %s %s" % (exec_name, infile, str(lam), outfile)
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
                theta = np.array(eval(v))
            else:
                raise Exception("error in parsing outfile of cpp executable")
    return theta

def learn(train, lam):
    n_users = get_user_count(train)
    assert(get_user_count(train) == get_user_count(val))
    theta = [1.0] * (n_users + 3)   # 1 alpha per user, 1 global alpha, theta0, theta1

    if (check_grad):
        print "Checking gradient before training.."
        ourgrad = np.linalg.norm(eprime_fn(np.array(theta), train, lam), ord = 2)
        print "gradient = ", ourgrad
        numerical = np.linalg.norm(scipy.optimize.approx_fprime(np.array(theta), e_fn, np.sqrt(np.finfo(np.float).eps), train, lam), ord = 2)
        print "numerical = ", numerical
        ratio = ourgrad / numerical
        print "ratio = ", ratio
        assert(abs(1 - ratio) < 0.00001)
        sys.exit(0)

    print "Training.."
    [theta, E_min, info] = scipy.optimize.fmin_l_bfgs_b(e_fn, theta, eprime_fn, args = (train, lam),  maxfun=100000, maxiter=100000, iprint=1, disp=1)
    return theta

if __name__ == "__main__":
    t1 = time.time()
    # prepare data set.. Run once and comment it out if running multiple times with same settings
    #infile = "endoMondo5000_workouts_condensed.gz"
    #infile = "../../data/all_workouts_train_and_val_condensed.gz"
    infile = "../../data/all_workouts_condensed.gz"
    #infile = "synth_user_model.gz"
    mode = "final"
    outfile = infile + mode + ".npz"
    #outfile = infile + ".npz"
    e_fn = E_pyx
    eprime_fn = Eprime_pyx
    check_grad = False

    #prepare(infile, outfile)

    lam = 0.0    # regularization
    if (len(sys.argv) == 2):
        lam = float(sys.argv[1])
    print "@lam1 = ", lam
    print "Loading data from file..", outfile
    data = np.load(outfile)
    train = data["train_set"]
    val = data["val_set"]

    theta = learn(train, lam)
    #theta = learn_cpp(train, lam)
    
    print "Computing predictions and statistics"
    [mse, var, fvu, r2] = compute_stats(train, theta)
    #print "\nStats for training data : \n# Examples = %d\nMSE = %f\nVariance = %f\nFVU = %f\nR2 = 1 - FVU = %f\n" % (train.shape[0],mse, var, fvu, r2)
    print "\n@Training Examples = %d,MSE = %f,Variance = %f,FVU = %f,R2 = 1 - FVU = %f\n" % (train.shape[0],mse, var, fvu, r2)
    [mse, var, fvu, r2] = compute_stats(val, theta)
    #print "\nStats for val data : \n# Examples = %d\nMSE = %f\nVariance = %f\nFVU = %f\nR2 = 1 - FVU = %f\n" % (val.shape[0],mse, var, fvu, r2)
    print "@Validation Examples = %d,MSE = %f,Variance = %f,FVU = %f,R2 = 1 - FVU = %f\n" % (val.shape[0], mse, var, fvu, r2)

    t2 = time.time()
    print "@Total time taken = ", t2 - t1

