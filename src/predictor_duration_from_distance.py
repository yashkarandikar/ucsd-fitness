import gzip
import utils
import numpy as np
import math
import matplotlib.pyplot as plt
from prepare_data_set import prepare_data_set
from plot_data import DataForPlot
from linear_reg import compute_stats, predictor_plots, residual_plots
import sys

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

    Xy = utils.remove_rows_by_condition(Xy, cols, lower_bounds, upper_bounds)
    [X, y] = utils.separate_Xy(Xy)
    return [X, y]

if __name__ == "__main__":
    # prepare data set.. Run once and comment it out if running multiple times with same settings
    infile = "../../data/all_workouts_train_and_val_condensed.gz"
    #infile = "endoMondo5000_workouts_condensed.gz"
    outfile = "train_val_duration_distance.npz"
    sport = "Running"
    x_params = ["Distance","pace(avg)","alt(avg)","hr(avg)","Total Ascent","Total Descent"]
    y_param = "Duration"
    missing_data_mode = "substitute"
    normalize = False
    prepare_data_set(infile = infile, sport = sport, x_params = x_params, y_param = y_param, outfile = outfile, missing_data_mode = missing_data_mode, normalize = normalize, outlier_remover = remove_outliers)
   
    # load data from file
    data = np.load(outfile)
    X = data["X"]
    y = data["y"]
    param_indices = data["param_indices"][()]   # [()] is required to convert numpy ndarray back to dictionary
    
    # split into training and validation
    [X_train, y_train, X_val, y_val] = utils.shuffle_and_split_Xy(X, y, fraction = 0.75)

    # extract relevant columns if not all columns need to be used - useful if you want to use certain features for training and then plot the residual errors against other features
    predictor_params = ["intercept","Distance"]   
    X_train_distance = utils.extract_columns_by_names(X_train, predictor_params, param_indices)
    X_val_distance = utils.extract_columns_by_names(X_val, predictor_params, param_indices)

    # train model
    theta,residuals,rank,s = np.linalg.lstsq(X_train_distance, y_train)

    # compute statistics on training set and validation set
    [mse, var, fvu, r2] = compute_stats(X_train_distance, y_train, theta)
    print "\nStats for training data : \n# Examples = %d\nMSE = %f\nVariance = %f\nFVU = %f\nR2 = 1 - FVU = %f\n" % (X_train.shape[0],mse, var, fvu, r2)
    [mse, var, fvu, r2] = compute_stats(X_val_distance, y_val, theta)
    print "Stats for validation data : \n# Examples = %d\nMSE = %f\nVariance = %f\nFVU = %f\nR2 = 1 - FVU = %f\n" % (X_val.shape[0], mse, var, fvu, r2)

    # residual plots and predictor plots
    residual_plots(X_train, X_train_distance, y_train, x_params, predictor_params, theta, param_indices)
    predictor_plots(X_train, y_train, x_params, y_param, predictor_params, param_indices)
    
    plt.show()

