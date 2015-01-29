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
#import pyximport; pyximport.install()
#import e_prime

def get_user_count(data):
    assert(type(data).__name__ == "list" or type(data).__name__ == "matrix" or type(data).__name__ == "ndarray")
    if (type(data).__name__ == "matrix" or type(data).__name__ == "ndarray"):
        return int(data[-1,0] + 1)   # since user numbers start from 0
    elif (type(data).__name__ == "list"):
        return int(data[-1][0] + 1)
    else:
        raise Exception("invalid type of data..")

def remove_outliers(data, params, param_indices):
    assert(type(data).__name__ == "matrix")
    N1 = data.shape[0]
    cols = []; lower_bounds = []; upper_bounds = []

    # remove rows distance < 0.01 mi
    c = param_indices["Distance"]; cols.append(c); lower_bounds.append(0.01); upper_bounds.append(float("inf"))

    # remove rows with duration < 0.01 hours
    c = param_indices["Duration"]; cols.append(c); lower_bounds.append(0.01 * 3600); upper_bounds.append(float("inf"))    # seconds

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

def shuffle_and_split_data_by_user(data, fraction = 0.5):
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
        end1 = int(math.ceil((fraction * float(n_u))))
        for p in perm[:end1]: mask[p] = 1
        for p in perm[end1:]: mask[p] = 2

    d1_indices = [i for i in range(0, N) if mask[i] == 1]
    d2_indices = [i for i in range(0, N) if mask[i] == 2]
    d1 = data[d1_indices, :]
    d2 = data[d2_indices, :]
    return [d1, d2]

def add_user_number_column(data, rare_user_threshold = 1):
    assert(type(data).__name__ == "matrix")
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

    print "Number of workouts discarded because very less data for that user : ", n_deleted
    return data

def convert_to_numbers(data):
    assert(type(data).__name__ == "list")
    for d in data:
        assert(len(d) == 3)
        d[0] = int(d[0])
        d[1] = float(d[1])
        d[2] = float(d[2])

def make_predictions(data, theta, param_indices):
    N = data.shape[0]
    T = [0.0] * N
    for i in range(0, N):
        u = int(data[i, param_indices["user_number"]])
        v = theta[u]
        d = data[i, param_indices["Distance"]]
        T[i] = v * d
    return [np.matrix([data[:, param_indices["Duration"]]]).T, np.matrix([T]).T]

def compute_stats(y, y_pred):
    mse = (np.square(y - y_pred)).mean()
    var = np.var(y)
    fvu = mse / var
    r2 = 1 - fvu
    return [mse, var,fvu, r2]

def string_list_to_dict(str_list):
    d = {}
    for p in str_list:
        d[p] = len(d.keys())
    return d

#def convert_pace_to_smi(data, param_indices):
    # convert pace from min/mi to s/mi
    #c = param_indices["pace(avg)"]
    #data[:, c] = 60.0 * data[:, c]
    
def prepare(infile, outfile):
    sport = "Running"
    params = ["user_id","Distance", "Duration"]
    param_indices = string_list_to_dict(params)
    
    print "Reading data.."
    data = read_data_as_lists(infile, sport, params)
    convert_to_numbers(data)   # convert from strings to numbers
    
    print "Converting data matrix to numpy format"
    data = np.matrix(data)

    print "Removing outliers.."
    data = remove_outliers(data, params, param_indices)
    
    print "Adding user numbers.."
    data = add_user_number_column(data, rare_user_threshold = 1)    # add a user number
    param_indices = string_list_to_dict(["user_number"] + params)
        
    print "Splitting data into training and validation"
    [d1, d2] = shuffle_and_split_data_by_user(data)
    
    print "Saving data to disk"
    np.savez(outfile, d1 = d1, d2 = d2, param_indices = param_indices)

def compute_avg_pace_for_users(data, param_indices):
    n_users = get_user_count(data)
    theta = [0.0] * n_users
    N = data.shape[0]
    d_ind = param_indices["Distance"]
    t_ind = param_indices["Duration"]
    i = 0
    while i < N:
        u = int(data[i, 0])
        total_d = 0.0
        total_t = 0.0
        while i < N and data[i, 0] == u:
            total_d += data[i, d_ind]
            total_t += data[i, t_ind]
            i += 1
        theta[u] = total_t / total_d
    return theta

if __name__ == "__main__":
    t1 = time.time()
    # prepare data set.. Run once and comment it out if running multiple times with same settings
    #infile = "endoMondo5000_workouts_condensed.gz"
    infile = "../../data/all_workouts_train_and_val_condensed.gz"
    #infile = "synth_baseline_model.gz"
    outfile = infile + ".npz"
    prepare(infile, outfile)
    data = np.load(outfile)
    train = data["d1"]
    val = data["d2"]
    param_indices = data["param_indices"][()]
    print param_indices
    n_users = get_user_count(train)
    assert(get_user_count(train) == get_user_count(val))
    print "Number of workouts (train) = ", train.shape[0]
    print "Number of workouts (val) = ", val.shape[0]
    print "Number of users = ", n_users

    # train and learn
    theta = compute_avg_pace_for_users(train, param_indices)

    # predict and evaluate
    [train_labels, train_pred] = make_predictions(train, theta, param_indices)
    [mse, var, fvu, r2] = compute_stats(train_labels, train_pred)
    print "\nStats for training data : \n# Examples = %d\nMSE = %f\nVariance = %f\nFVU = %f\nR2 = 1 - FVU = %f\n" % (train.shape[0],mse, var, fvu, r2)
    
    [val_labels, val_pred] = make_predictions(val, theta, param_indices)
    [mse, var, fvu, r2] = compute_stats(val_labels, val_pred)
    print "\nStats for val data : \n# Examples = %d\nMSE = %f\nVariance = %f\nFVU = %f\nR2 = 1 - FVU = %f\n" % (val.shape[0],mse, var, fvu, r2)
    t2 = time.time()
    
    print "Total time taken = ", t2 - t1
