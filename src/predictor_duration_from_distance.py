import gzip
import utils
import numpy as np
import math
import matplotlib.pyplot as plt
from prepare_data_set import prepare_data_set
from plot_data import DataForPlot
from linear_reg import compute_stats
import sys

def remove_rows_by_condition(m, cols, lower_bounds, upper_bounds):
    assert(len(cols) == len(lower_bounds) and len(lower_bounds) == len(upper_bounds))
    n = len(cols)
    for i in range(0, n):
        c = cols[i]
        m = utils.sort_matrix_by_col(m, c)
        l = lower_bounds[i]; u = upper_bounds[i]
        i = np.searchsorted(m[:, c].A1, lower_bounds[i])
        j = np.searchsorted(m[:, c].A1, u)
        m = m[i:j, :]
    return m

def shuffle_and_split_data(X, y, fraction):
    Xy = utils.combine_Xy(X, y)
    np.random.shuffle(Xy)
    end1 = fraction * Xy.shape[0]
    Xy_1 = Xy[:end1, :]
    Xy_2 = Xy[end1:, :]
    [X_1, y_1] = utils.separate_Xy(Xy_1)
    [X_2, y_2] = utils.separate_Xy(Xy_2)
    return [X_1, y_1, X_2, y_2]

"""
def param_to_col(p, x_params, missing_data_mode):
    c = 0
    while c < len(x_params):
        if (x_params[c] == p):
            break
        c += 1
    assert(c < len(x_params))
    assert(x_params[c] == p)
    if (missing_data_mode == "substitute"):
        c = 1 + 2 * c
    return c
"""

def remove_outliers(X, y, x_params, y_param, missing_data_mode, param_indices):
    print "Removing outliers.."
    Xy = utils.combine_Xy(X, y)
    params = x_params + [y_param]
    assert(missing_data_mode == "ignore" or missing_data_mode == "substitute")
    if (missing_data_mode == "ignore"):
        assert(len(params) == Xy.shape[1])
    else:
        assert(2.0 * len(params) - 1 == Xy.shape[1])
    
    cols = []; lower_bounds = []; upper_bounds = []

    # remove rows distance < 0.1 mi and > 80 mi
    #c = param_to_col("Distance", x_params, missing_data_mode)
    c = param_indices["Distance"]
    cols.append(c); lower_bounds.append(0.1); upper_bounds.append(80.0)
    
    # remove rows with duration < 0.1 and > 36000
    cols.append(Xy.shape[1] - 1)    # Duration
    lower_bounds.append(0.1); upper_bounds.append(36000.0)

    Xy = remove_rows_by_condition(Xy, cols, lower_bounds, upper_bounds)
    [X, y] = utils.separate_Xy(Xy)
    return [X, y]

    """
    # remove rows distance < 0.1 mi and > 80 mi
    Xy = utils.sort_matrix_by_col(Xy, 0)
    print "Xy shape = ", Xy.shape
    i = np.searchsorted(Xy[:, 0].A1, 0.1)
    j = np.searchsorted(Xy[:, 0].A1, 80.0)
    Xy = Xy[i:j, :]

    # remove rows with duration < 0.1 and > 36000
    Xy = utils.sort_matrix_by_col(Xy, 1)
    i = np.searchsorted(Xy[:, -1].A1, 0.1)
    j = np.searchsorted(Xy[:, -1].A1, 36000.0)
    Xy = Xy[i:j, :]
    """

def extract_columns_by_names(m, params, param_indices):
    p = params[0]
    m_new = None
    if (param_indices.has_key(p + "_present")):
        m_new = np.matrix(m[:, param_indices[p + "_present"]]).T
        m_new = np.concatenate((m_new, np.matrix(m[:, param_indices[p]]).T), axis = 1)
    else:
        m_new = np.matrix(m[:, param_indices[p]]).T
    
    for i in range(1, len(params)):
        p = params[i]
        i = param_indices[p]
        if (param_indices.has_key(p + "_present")):
            m_new = np.concatenate((m_new, np.matrix(m[:, param_indices[p + "_present"]]).T), axis = 1)
        m_new = np.concatenate((m_new, np.matrix(m[:, i]).T), axis = 1)
    assert(m_new.shape[0] == m.shape[0])
    return m_new

def residual_plots(X_full, X_predictor, y, x_params, predictor_params, param_indices):
    # residual plots
    errors = y - X_predictor.dot(theta)
    print errors.shape
    other_params = [p for p in x_params if (p not in predictor_params) ]
    for p in other_params:
        c = param_indices[p]
        plt.figure()
        plt.plot(X_full[:, c], errors, 'o')
        plt.xlabel(p)
        plt.ylabel('error')

def predictor_plots(X, y, x_params, y_param, predictor_params, param_indices):
    # plot with predictor params
    for p in predictor_params:
        c = param_indices[p]
        plt.figure()
        plt.plot(X[:, c], y, 'o')
        plt.xlabel(p)
        plt.ylabel(y_param)

if __name__ == "__main__":
    #train_base = "train_duration_distance"
    #val_base = "val_duration_distance"
    infile = "../../data/all_workouts_train_and_val_condensed.gz"
    #infile = "endoMondo5000_workouts_condensed.gz"
    outfile = "train_val_duration_distance.npz"
    sport = "Running"
    x_params = ["Distance","pace(avg)","alt(avg)","hr(avg)","Total Ascent","Total Descent"]
    y_param = "Duration"
    missing_data_mode = "substitute"
    normalize = False
    prepare_data_set(infile = infile, sport = sport, x_params = x_params, y_param = y_param, outfile = outfile, missing_data_mode = missing_data_mode, normalize = normalize, outlier_remover = remove_outliers)
    #prepare_data_set(infile = "../../data/all_workouts_validation_condensed.gz", sport = "Running", x_params = ["Distance"], y_param = "Duration", outfile_base=val_base, missing_data_mode = "ignore", normalize = False, outlier_remover = remove_outliers)

    #file_X_train = train_base + "X.npy"
    #file_y_train = train_base + "y.npy"
    #file_X_val = val_base + "X.npy"
    #file_y_val = val_base + "y.npy"

    # read data from files
    #X_train = np.load(file_X_train)
    #y_train = np.load(file_y_train)
    #if (file_X_val is not None and file_y_val is not None):
    #    X_val = np.load(file_X_val)
    #    y_val = np.load(file_y_val)

    #X = np.load(outfile_base + "X.npy");  y = np.load(outfile_base + "y.npy")
    data = np.load(outfile)
    X = data["X"]
    y = data["y"]
    param_indices = data["param_indices"][()]   # [()] is required to convert numpy ndarray back to dictionary
    
    [X_train, y_train, X_val, y_val] = shuffle_and_split_data(X, y, fraction = 0.75)

    predictor_params = ["intercept","Distance"]   
    X_train_distance = extract_columns_by_names(X_train, predictor_params, param_indices)
    X_val_distance = extract_columns_by_names(X_val, predictor_params, param_indices)

    # train model
    theta,residuals,rank,s = np.linalg.lstsq(X_train_distance, y_train)
    #print residuals

    # compute statistics on training set and validation set
    [mse, var, fvu, r2] = compute_stats(X_train_distance, y_train, theta)
    print "\nStats for training data : \n# Examples = %d\nMSE = %f\nVariance = %f\nFVU = %f\nR2 = 1 - FVU = %f\n" % (X_train.shape[0],mse, var, fvu, r2)
    [mse, var, fvu, r2] = compute_stats(X_val_distance, y_val, theta)
    print "Stats for validation data : \n# Examples = %d\nMSE = %f\nVariance = %f\nFVU = %f\nR2 = 1 - FVU = %f\n" % (X_val.shape[0], mse, var, fvu, r2)

    # residual plots
    residual_plots(X_train, X_train_distance, y_train, x_params, predictor_params, param_indices)
    predictor_plots(X_train, y_train, x_params, y_param, predictor_params, param_indices)
    
    #c_distance = param_to_col("Distance", x_params, missing_data_mode)
    #c_distance = param_indices['Distance']
    #plt.plot(X[:, c_distance], y_train, 'o')
    
    plt.show()

