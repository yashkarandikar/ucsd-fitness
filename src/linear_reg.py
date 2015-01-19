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

def residual_plots(X_full, X_predictor, y, x_params, predictor_params, theta, param_indices):
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

