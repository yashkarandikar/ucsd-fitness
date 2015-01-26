import gzip
import utils
import numpy as np
import math
import matplotlib.pyplot as plt
from prepare_data_set import prepare_data_set, read_data_as_lists
from plot_data import DataForPlot
#from linear_reg import compute_stats, predictor_plots, residual_plots
import sys
from unit import Unit
import scipy.optimize
import time

def remove_outliers(X, y, x_params, y_param, missing_data_mode, param_indices):
    print "Removing outliers.."
    N1 = X.shape[0]
    Xy = utils.combine_Xy(X, y)
    params = x_params + [y_param]
    assert(missing_data_mode == "ignore" or missing_data_mode == "substitute")
    if (missing_data_mode == "ignore"):
        assert(len(params) == Xy.shape[1])
    else:
        assert(2.0 * len(params) - 1 == Xy.shape[1])
    
    cols = []; lower_bounds = []; upper_bounds = []

    # remove rows distance < 0.1 mi and > 80 mi
    c = param_indices["Distance"]; cols.append(c); lower_bounds.append(0.1); upper_bounds.append(80.0)

    # remove rows with duration < 0.1 and > 36000
    cols.append(Xy.shape[1] - 1)    # Duration
    lower_bounds.append(0.1); upper_bounds.append(36000.0)

    # remove rows with other parameters < 0.1
    #c = param_indices["pace(avg)"]; cols.append(c); lower_bounds.append(0.1); upper_bounds.append(float("inf"))
    c = param_indices["Total Ascent"]; cols.append(c); lower_bounds.append(float("-inf")); upper_bounds.append(15000)
    c = param_indices["Total Descent"]; cols.append(c); lower_bounds.append(float("-inf")); upper_bounds.append(15000)
    c = param_indices["alt(avg)"]; cols.append(c); lower_bounds.append(-1000); upper_bounds.append(30000)

    Xy = utils.remove_rows_by_condition(Xy, cols, lower_bounds, upper_bounds)
    [X, y] = utils.separate_Xy(Xy)
    N2 = X.shape[0]
    print "%d rows removed during outlier removal.." % (N1 - N2)
    return [X, y]

def is_sorted(data):
    N = len(data)
    for i in range(0, N - 1):
        if (data[i, 1] > data[i+1, 1]):
            return False
    return True

def E(theta, data):
    # error function to be minimized
    # assumes data has 4 columns : user_id, user_number, distance, duration and that it is sorted
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
    print "E = ", e
    return e

"""
def E_vec(theta, data, workout_count_for_user):
    # error function to be minimized
    # assumes data has 4 columns : user_id, user_number, distance, duration and that it is sorted
    
    # append column of alpha and scale distances column by alphas
    n_users = len(workout_count_for_user)
    n_workouts = data.shape[0]
    alpha_temp = theta[:n_users]
    alpha = []
    for i in range(0, n_users):
        alpha = alpha + [alpha_temp[i]] * workout_count_for_user[i]
    alpha = np.matrix([alpha]).T
    data_aug = np.concatenate((alpha, data), axis = 1)
    temp = np.concatenate((np.ones((n_workouts, 3)), alpha, np.ones((n_workouts, 1))), axis = 1)
    data_aug = np.multiply(data_aug, temp)

    theta_0 = theta[-2]
    theta_1 = theta[-1]
    theta_vec = np.matrix([[theta_0], [0.0], [0.0], [1.0], [0.0]])
    diff = data_aug.dot(theta_vec) - data_aug[:, -1]
    assert(diff.shape[0] == n_workouts)
    assert(diff.shape[1] == 1)
    e = ((diff.T).dot(diff))[0, 0]
    print "E = ", e
    return e
"""

def Eprime(theta, data):
    N = len(data)
    n_users = data[-1, 0]
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
    return dE

def add_user_number_column(data):
    data.sort(key=lambda x: x[0])
    n = len(data)
    uin = 0
    i = 0
    #workout_count_for_user = [0]
    while i < n:
        u = data[i][0]
        #workout_count_for_user.append(0)
        while i < n and data[i][0] == u:
            data[i] = [uin] + data[i]
            i += 1
            #workout_count_for_user[uin] += 1
        uin += 1
    #return workout_count_for_user

def convert_to_strings(data):
    for d in data:
        assert(len(d) == 3)
        d[0] = int(d[0])
        d[1] = float(d[1])
        d[2] = float(d[2])

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

def split_data_by_user(data, fraction = 0.5):
    # assumes data is numpy matrix form
    assert(type(data).__name__ == "matrix")
    d1 = None; d2 = None;
    i = 0
    N = len(data)
    randomState = np.random.RandomState(seed = 12345)
    while i < N:
        u = data[i, 0]
        data_u = data[i, :]
        i += 1
        # get all rows for this user
        while i < N and data[i, 0] == u:
            data_u = np.concatenate((data_u, data[i]), axis = 0)
            #data_u = data_u + data[i]
            i += 1
        if len(data_u) > 1:    # discard users with only 1 workout
            # shuffle and split
            [m1, m2] = utils.shuffle_and_split_mat_rows(data_u, fraction = fraction, randomState = randomState)
            #[m1, m2] = utils.shuffle_and_split_lists(data_u, fraction = fraction, seed = 12345)
            if (d1 is None and d2 is None):
                d1 = m1; d2 = m2
            else:
                d1 = np.concatenate((d1, m1), axis = 0)
                d2 = np.concatenate((d2, m2), axis = 0)
            #d1 = d1 + m1
            #d2 = d2 + m2
    print len(d1)
    print len(d2)
    return [d1, d2]

def prepare(infile, outfile):
    sport = "Running"
    params = ["user_id","Distance", "Duration"]
    data = read_data_as_lists(infile, sport, params)
    print "Converting strings to numbers.."
    convert_to_strings(data)   # convert from strings to numbers
    print "Sorting data by users.."
    data.sort(key=lambda x: x[0])   # sort by user ID
    print "Adding user numbers.."
    add_user_number_column(data)    # add a user number 
    print "Converting data matrix to numpy format"
    data = np.matrix(data)
    print "Splitting data into training and validation"
    [d1, d2] = split_data_by_user(data)
    print "Saving data to disk"
    np.savez(outfile, d1 = d1, d2 = d2)

if __name__ == "__main__":
    # prepare data set.. Run once and comment it out if running multiple times with same settings
    infile = "endoMondo5000_workouts_condensed.gz"
    #infile = "../../data/all_workouts_train_and_val_condensed.gz"
    outfile = "train_val_distance_user.npz"
    prepare(infile, outfile)
    data = np.load(outfile)
    train = data["d1"]
    val = data["d2"]    
    n_users = train[-1, 0]
    print "Number of users = ", n_users
    theta = [1.0] * (n_users + 2)
    [theta, E_min, info] = scipy.optimize.fmin_l_bfgs_b(E, theta, Eprime, args = (train, ), maxfun=100)
    print "length of theta vector = ", len(theta)
    print info
    [mse, var, fvu, r2] = compute_stats(train, theta)
    print "\nStats for training data : \n# Examples = %d\nMSE = %f\nVariance = %f\nFVU = %f\nR2 = 1 - FVU = %f\n" % (train.shape[0],mse, var, fvu, r2)
    [mse, var, fvu, r2] = compute_stats(val, theta)
    print "\nStats for val data : \n# Examples = %d\nMSE = %f\nVariance = %f\nFVU = %f\nR2 = 1 - FVU = %f\n" % (val.shape[0],mse, var, fvu, r2)

    #y_param = "Duration"
    #missing_data_mode = "substitute"
    #normalize = False
    #split_fraction = 0.75
    #outlier_remover = remove_outliers
    #randomState = np.random.RandomState(seed = 12345)
    #prepare_data_set(infile = infile, sport = sport, x_params = x_params, y_param = y_param, outfile = outfile, missing_data_mode = missing_data_mode, normalize = normalize, outlier_remover = outlier_remover, split_fraction = split_fraction)
   
    # load data from file
    #data = np.load(outfile)
    #X_train = data["X1"]
    #y_train = data["y1"]
    #X_val = data["X2"]
    #y_val = data["y2"]
    #param_indices = data["param_indices"][()]   # [()] is required to convert numpy ndarray back to dictionary
    
    # extract relevant columns if not all columns need to be used - useful if you want to use certain features for training and then plot the residual errors against other features and train model
    #predictor_params = ["intercept","Distance"]   
    #X_train_distance = utils.extract_columns_by_names(X_train, predictor_params, param_indices)
    #X_val_distance = utils.extract_columns_by_names(X_val, predictor_params, param_indices)
    
    #print "theta = ", theta

    # compute statistics on training set and validation set
    #[mse, var, fvu, r2] = compute_stats(data, theta)
    #print "\nStats for training data : \n# Examples = %d\nMSE = %f\nVariance = %f\nFVU = %f\nR2 = 1 - FVU = %f\n" % (data.shape[0],mse, var, fvu, r2)
    #[mse, var, fvu, r2] = compute_stats(X_val_distance, y_val, theta)
    #print "Stats for validation data : \n# Examples = %d\nMSE = %f\nVariance = %f\nFVU = %f\nR2 = 1 - FVU = %f\n" % (X_val.shape[0], mse, var, fvu, r2)
