import gzip
import utils
import numpy as np
import math
import matplotlib.pyplot as plt
from prepare_data_set import prepare_data_set
from plot_data import DataForPlot
from linear_reg import compute_stats, predictor_plots, residual_plots
import sys
from unit import Unit

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

    # remove rows distance < 0.01 mi 
    c = param_indices["Distance"]; cols.append(c); lower_bounds.append(0.01); upper_bounds.append(float("inf"))

    # remove rows with duration < 0.01 hours
    cols.append(Xy.shape[1] - 1)    # Duration
    lower_bounds.append(0.01 * 3600); upper_bounds.append(float("inf"))

    # remove rows with other parameters < 0.1
    #c = param_indices["pace(avg)"]; cols.append(c); lower_bounds.append(0.1); upper_bounds.append(float("inf"))
    #c = param_indices["Total Ascent"]; cols.append(c); lower_bounds.append(float("-inf")); upper_bounds.append(15000)
    #c = param_indices["Total Descent"]; cols.append(c); lower_bounds.append(float("-inf")); upper_bounds.append(15000)
    #c = param_indices["alt(avg)"]; cols.append(c); lower_bounds.append(-1000); upper_bounds.append(30000)

    Xy = utils.remove_rows_by_condition(Xy, cols, lower_bounds, upper_bounds)
    [X, y] = utils.separate_Xy(Xy)
    N2 = X.shape[0]
    print "%d rows removed during outlier removal.." % (N1 - N2)
    return [X, y]

if __name__ == "__main__":
    # prepare data set.. Run once and comment it out if running multiple times with same settings
    infile = "../../data/all_workouts_train_and_val_condensed.gz"
    #infile = "endoMondo5000_workouts_condensed.gz"
    outfile = "train_val_duration_distance.npz"
    sport = "Running"
    x_params = ["Distance"]
    y_param = "Duration"
    missing_data_mode = "ignore"
    normalize = False
    split_fraction = 0.5
    outlier_remover = remove_outliers
    #randomState = np.random.RandomState(seed = 12345)
    prepare_data_set(infile = infile, sport = sport, x_params = x_params, y_param = y_param, outfile = outfile, missing_data_mode = missing_data_mode, normalize = normalize, outlier_remover = outlier_remover, split_fraction = split_fraction)
   
    # load data from file
    data = np.load(outfile)
    X_train = data["X1"]
    y_train = data["y1"]
    X_val = data["X2"]
    y_val = data["y2"]
    param_indices = data["param_indices"][()]   # [()] is required to convert numpy ndarray back to dictionary
    
    # extract relevant columns if not all columns need to be used - useful if you want to use certain features for training and then plot the residual errors against other features and train model
    #predictor_params = ["intercept","Distance"]   
    #X_train_distance = utils.extract_columns_by_names(X_train, predictor_params, param_indices)
    #X_val_distance = utils.extract_columns_by_names(X_val, predictor_params, param_indices)
    X_train_distance = X_train
    X_val_distance = X_val
    theta,residuals,rank,s = np.linalg.lstsq(X_train_distance, y_train)

    print "theta = ", theta

    # compute statistics on training set and validation set
    [mse, var, fvu, r2] = compute_stats(X_train_distance, y_train, theta)
    print "\nStats for training data : \n# Examples = %d\nMSE = %f\nVariance = %f\nFVU = %f\nR2 = 1 - FVU = %f\n" % (X_train.shape[0],mse, var, fvu, r2)
    [mse, var, fvu, r2] = compute_stats(X_val_distance, y_val, theta)
    print "Stats for validation data : \n# Examples = %d\nMSE = %f\nVariance = %f\nFVU = %f\nR2 = 1 - FVU = %f\n" % (X_val.shape[0], mse, var, fvu, r2)

    # residual plots and predictor plots
    """
    errors = y_train - X_train_distance.dot(theta)

    plt.figure()

    plt.subplot(3,2,0)
    p = "alt(avg)"
    alt = DataForPlot(sport, p, "error", xvals = X_train[:, param_indices[p]], yvals = errors)
    alt.plot_simple(x_range = [-1000, 10000])
    plt.ylabel("Error in duration(" + Unit.get("Duration") + ")")
    
    plt.subplot(3,2,1)
    p = "hr(avg)"
    alt = DataForPlot(sport, p, "error", xvals = X_train[:, param_indices[p]], yvals = errors)
    alt.plot_simple(x_range = [0, 250])
    plt.ylabel("Error in duration(" + Unit.get("Duration") + ")")
    
    plt.subplot(3,2,2)
    p = "Total Ascent"
    alt = DataForPlot(sport, p, "error", xvals = X_train[:, param_indices[p]], yvals = errors)
    alt.plot_simple(x_range = [0, 20000])
    plt.ylabel("Error in duration(" + Unit.get("Duration") + ")")
    
    plt.subplot(3,2,3)
    p = "Total Descent"
    alt = DataForPlot(sport, p, "error", xvals = X_train[:, param_indices[p]], yvals = errors)
    alt.plot_simple(x_range = [0, 20000])
    plt.ylabel("Error in duration(" + Unit.get("Duration") + ")")

    plt.subplot(3,2,4)
    p = "pace(avg)"
    alt = DataForPlot(sport, p, "error", xvals = X_train[:, param_indices[p]], yvals = errors)
    alt.plot_simple(x_range = [0, 40])
    plt.ylabel("Error in duration(" + Unit.get("Duration") + ")")
    
    #predictor_plots(X_train, y_train, x_params, y_param, predictor_params, param_indices)
    plt.tight_layout()
    plt.show()
    """

