import gzip
import utils
import numpy as np
import math
import matplotlib.pyplot as plt
from prepare_data_set import prepare_data_set
from plot_data import DataForPlot
from linear_reg import compute_stats

def remove_outliers(X, y):
    print "Removing outliers.."
    Xy = utils.combine_Xy(X, y)

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

    [X, y] = utils.separate_Xy(Xy)
    return [X, y]

if __name__ == "__main__":
    train_base = "train_duration_distance"
    val_base = "val_duration_distance"
    prepare_data_set(infile = "../../data/all_workouts_train_condensed.gz", sport = "Running", x_params = ["Distance"], y_param = "Duration", outfile_base=train_base, missing_data_mode = "ignore", normalize = False, outlier_remover = remove_outliers)
    prepare_data_set(infile = "../../data/all_workouts_validation_condensed.gz", sport = "Running", x_params = ["Distance"], y_param = "Duration", outfile_base=val_base, missing_data_mode = "ignore", normalize = False, outlier_remover = remove_outliers)

    file_X_train = train_base + "X.npy"
    file_y_train = train_base + "y.npy"
    file_X_val = val_base + "X.npy"
    file_y_val = val_base + "y.npy"

    # read data from files
    X_train = np.load(file_X_train)
    y_train = np.load(file_y_train)
    if (file_X_val is not None and file_y_val is not None):
        X_val = np.load(file_X_val)
        y_val = np.load(file_y_val)

    # train model
    theta,residuals,rank,s = np.linalg.lstsq(X_train, y_train)
    #print residuals

    # compute statistics on training set and validation set
    [mse, var, fvu, r2] = compute_stats(X_train, y_train, theta)
    print "\nStats for training data : \n# Examples = %d\nMSE = %f\nVariance = %f\nFVU = %f\nR2 = 1 - FVU = %f\n" % (X_train.shape[0],mse, var, fvu, r2)
    if (file_X_val is not None and file_y_val is not None):
        [mse, var, fvu, r2] = compute_stats(X_val, y_val, theta)
        print "Stats for validation data : \n# Examples = %d\nMSE = %f\nVariance = %f\nFVU = %f\nR2 = 1 - FVU = %f\n" % (X_val.shape[0], mse, var, fvu, r2)

    plt.plot(X_train[:, 1], y_train, 'o')
    plt.show()
