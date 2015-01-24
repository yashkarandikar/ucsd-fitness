import gzip
import utils
import numpy as np
import math
import matplotlib.pyplot as plt
from prepare_data_set import prepare_data_set, read_data_as_lists
from plot_data import DataForPlot
from linear_reg import compute_stats, predictor_plots, residual_plots
import sys
from unit import Unit
import scipy.optimize

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
    #assert(is_sorted(data))
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
    uin = 1
    i = 0
    while i < n:
        u = data[i][0]
        while i < n and data[i][0] == u:
            data[i] = [uin] + data[i]
            i += 1
        uin += 1

def convert_to_ints(data):
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
        u = data[i][0]
        alpha = theta[u]
        d = data[i][2]
        t[i] = data[i][3]
        tprime[i] = math.pow(alpha * (theta_0 + theta_1 * d) - t, 2)
    mse = (np.square(t - tprime)).mean()
    var = np.var(t)
    fvu = mse / var
    r2 = 1 - fvu
    return [mse, var,fvu, r2]


if __name__ == "__main__":
    # prepare data set.. Run once and comment it out if running multiple times with same settings
    #infile = "../../data/all_workouts_train_and_val_condensed.gz"
    infile = "endoMondo5000_workouts_condensed.gz"
    outfile = "train_val_duration_distance_user.npz"
    sport = "Running"
    params = ["user_id","Distance", "Duration"]
    data = read_data_as_lists(infile, sport, params)
    convert_to_ints(data)
    data.sort(key=lambda x: x[0])
    add_user_number_column(data)
    data = np.matrix(data)
    print data
    
    n_users = data[-1, 0]
    print "Number of users = ", n_users
    #theta = np.array([0.0] * (n_users + 2))
    theta = [1.0] * (n_users + 2)
    print E(theta, data)
    print Eprime(theta, data)
    [theta, E_min, info] = scipy.optimize.fmin_l_bfgs_b(E, theta, Eprime, args = (data, ))
    print info

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
    [mse, var, fvu, r2] = compute_stats(data, theta)
    print "\nStats for training data : \n# Examples = %d\nMSE = %f\nVariance = %f\nFVU = %f\nR2 = 1 - FVU = %f\n" % (X_train.shape[0],mse, var, fvu, r2)
    #[mse, var, fvu, r2] = compute_stats(X_val_distance, y_val, theta)
    #print "Stats for validation data : \n# Examples = %d\nMSE = %f\nVariance = %f\nFVU = %f\nR2 = 1 - FVU = %f\n" % (X_val.shape[0], mse, var, fvu, r2)
