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
