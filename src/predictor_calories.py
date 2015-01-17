import gzip
import utils
import numpy as np
import math
import matplotlib.pyplot as plt

def compute_stats(X, Y, theta):
    Y_predicted = X.dot(theta)
    mse = (np.square(Y_predicted - Y)).mean()
    var = np.var(Y)
    fvu = mse / var
    r2  = 1 - fvu
    return [mse, var, fvu, r2]

def remove_outliers(X, y):
    Xy = utils.combine_Xy(X, y)

    # remove rows distance < 0.1 mi and > 100 mi
    Xy = utils.sort_matrix_by_col(Xy, 1)
    i = np.searchsorted(Xy[:, 1], 0.1)
    j = np.searchsorted(Xy[:, 1], 100.0)
    Xy = Xy[i:j, :]

    # remove rows with calories < 0.1 and > 5000
    Xy = utils.sort_matrix_by_col(Xy, 2)
    i = np.searchsorted(Xy[:, -1], 0.1)
    j = np.searchsorted(Xy[:, -1], 5000.0)
    Xy = Xy[i:j, :]

    [X, y] = utils.separate_Xy(Xy)
    return [X, y]

if __name__ == "__main__":
    file_X_train = "train_calories_distanceX.npy"
    file_y_train = "train_calories_distancey.npy"
    file_X_val = "val_calories_distanceX.npy"
    file_y_val = "val_calories_distancey.npy"

    # read data from files
    X_train = np.load(file_X_train)
    y_train = np.load(file_y_train)
    if (file_X_val is not None and file_y_val is not None):
        X_val = np.load(file_X_val)
        y_val = np.load(file_y_val)
    print X_train.shape
    print y_train.shape

    # remove outliers
    [X_train, y_train] = remove_outliers(X_train, y_train)
    [X_val, y_val] = remove_outliers(X_val, y_val)
    print X_train.shape
    print y_train.shape
    print X_train

    # train model
    theta,residuals,rank,s = np.linalg.lstsq(X_train, y_train)
    #print residuals

    # compute statistics on training set and validation set
    [mse, var, fvu, r2] = compute_stats(X_train, y_train, theta)
    print "\nStats for training data : \nMSE = %f\nVariance = %f\nFVU = %f\nR2 = 1 - FVU = %f\n" % (mse, var, fvu, r2)
    if (file_X_val is not None and file_y_val is not None):
        [mse, var, fvu, r2] = compute_stats(X_val, y_val, theta)
        print "Stats for validation data : \nMSE = %f\nVariance = %f\nFVU = %f\nR2 = 1 - FVU = %f\n" % (mse, var, fvu, r2)

    plt.plot(X_train[:, 1], y_train, 'o')
    plt.show()

    
