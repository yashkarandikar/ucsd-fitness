import gzip
import utils
import numpy as np
import math

def compute_stats(X, Y, theta):
    Y_predicted = X.dot(theta)
    mse = (np.square(Y_predicted - Y)).mean()
    var = np.var(Y)
    fvu = mse / var
    r2  = 1 - fvu
    return [mse, var, fvu, r2]

def linear_reg(file_X_train, file_Y_train, file_X_val, file_Y_val, file_X_test, file_Y_test):
    # read data from files
    X_train = np.load(file_X_train)
    Y_train = np.load(file_Y_train)
    X_val = np.load(file_X_val)
    Y_val = np.load(file_Y_val)
    X_test = np.load(file_X_test)
    Y_test = np.load(file_Y_test)
        
    # train model
    theta,residuals,rank,s = np.linalg.lstsq(X_train, Y_train)
    print theta
    print residuals

    # compute statistics on training set and validation set
    [mse, var, fvu, r2] = compute_stats(X_train, Y_train, theta)
    print "\nStats for training data : \nMSE = %f\nVariance = %f\nFVU = %f\nR2 = 1 - FVU = %f\n" % (mse, var, fvu, r2)
    [mse, var, fvu, r2] = compute_stats(X_val, Y_val, theta)
    print "Stats for validation data : \nMSE = %f\nVariance = %f\nFVU = %f\nR2 = 1 - FVU = %f\n" % (mse, var, fvu, r2)

if __name__ == "__main__":
    linear_reg("X_train.npy","Y_train.npy","X_val.npy","Y_val.npy","X_test.npy","Y_test.npy")
