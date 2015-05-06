import gzip
import utils
import numpy as np
import math
import matplotlib.pyplot as plt
from plot_data import DataForPlot
import sys
from unit import Unit
import scipy.optimize
import time
import random
import sys
import os
from param_formatter import ParamFormatter
import psutil as ps
import statsmodels.api as sm
from sklearn import linear_model
from datetime import datetime

def get_workout_count(data):
    assert(type(data).__name__ == "list" or type(data).__name__ == "matrix" or type(data).__name__ == "ndarray")
    if (type(data).__name__ == "matrix" or type(data).__name__ == "ndarray"):
        return int(data[-1,0] + 1)   # since user numbers start from 0
    elif (type(data).__name__ == "list"):
        return int(data[-1][0] + 1)
    else:
        raise Exception("invalid type of data..")

def is_sorted(data):
    N = len(data)
    for i in range(0, N - 1):
        if (data[i, 1] > data[i+1, 1]):
            return False
    return True

def make_predictions(all_X, all_models):
    W = len(all_X)
    all_y = [0.0] * W
    for w in xrange(0, W):
        all_y[w] = all_models[w].predict(all_X[w])
        for p in all_y[w]:
            if (p > 400.0):
                print p
    pred = np.matrix(np.concatenate(tuple(all_y))).T
    #for i in xrange(0, pred.shape[0]):
    #    pred[i, 0] = pred[0, 0]
    return pred

def predict_next_ARMA(all_models, start_all, end_all):
    W = len(all_models)
    all_y = [0.0] * W
    for w in xrange(0, W):
        m = all_models[w]
        all_y[w] = m.predict(start = start_all[w], end = end_all[w])
        #all_y[w] = m.predict()
        if (w % 10000 == 0):
            print "Done %d workouts, out of %d" % (w, W)

    #pred = np.matrix(np.concatenate(tuple(all_y))).T
    return all_y


def predict_next(all_last_E, all_models, next_n, E):
    W = len(all_models)
    assert(len(all_last_E) == W)
    assert(len(next_n) == W)
    all_y = [0.0] * W
    for w in xrange(0, W):
        m = all_models[w]
        n = next_n[w]
        all_y[w] = [0.0] * n
        x = [0.0] * (n + E)
        x[:E] = all_last_E[w]
        for i in xrange(0, n):
            #p = m.predict(np.matrix([x]))[0]
            v = x[i:i+E]
            p = m.predict(np.insert(v, 0, 1.0))
            if (p > 400.0):
                print "ALERT : w = ", w
                #print all_last_E[w]
                #print v
            all_y[w][i] = p
            x[E + i] = p
            #x.append(p)
            #x = x[1:]
            #x = np.append(x, p)
        if (w % 10000 == 0):
            print "Done %d workouts, out of %d" % (w, W)

    #pred = np.matrix(np.concatenate(tuple(all_y))).T
    return all_y

def compute_stats(t_actual, t_pred):
    #assert(t_actual.shape[1] == 1 and t_pred.shape[1] == 1)
    assert(t_actual.shape == t_pred.shape)
    mse = (np.square(t_actual - t_pred)).mean()
    errors = (t_actual - t_pred).A1
    var = np.var(t_actual)
    fvu = mse / var
    r2 = 1 - fvu
    return [mse, var,fvu, r2, errors]

def string_list_to_dict(str_list):
    d = {}
    for p in str_list:
        d[p] = len(d.keys())
    return d

def convert_sec_to_hours(data, param_indices):
    c = param_indices["duration"]
    data[:, c] = data[:, c] / 3600.0

def normalize(data, cols):
    F = data.shape[1]
    scale_factors = [1.0] * F
    for c in cols:
        scale_factors[c] = np.max(data[:, c])
        data[:, c] /= scale_factors[c]
    return scale_factors

def organize_for_learning(vals, E):
    P = E
    X = np.zeros((len(vals) - P, P + 1))
    y = np.zeros(len(vals) - P)
    for p in xrange(0, len(vals) - P):
        X[p, 1:] = vals[p : p + P]
        y[p] = vals[p + P]
    X[:, 0] = 1.0
    return [X, y]

def extract_col(data, col):
    if (type(data).__name__ == "matrix"):
        return data[:, col].A1
    else:
        return data[:, col]

def organize(data, E, param_indices):
    print "Organizing data.."
    # organizes data as a matrix for linear regression    
    hr_ind = param_indices["hr"]
    W = get_workout_count(data)
    
    wins = np.array(range(0, W))
    w_indices = list(np.searchsorted(extract_col(data, 0), wins))
    w_indices.append(data.shape[0])

    all_X = [0.0] * W
    all_y = [0.0] * W

    for w in xrange(0, W):
        start, end = w_indices[w : w + 2]
        hr = data[start:end, hr_ind]
        X, y = organize_for_learning(hr, E)
        all_X[w] = X
        all_y[w] = y

        if (w % 10000 == 0):
            print "Done %d out of %d" % (w, W)
   
    return all_X, all_y

def organize_data(train_set, val_set, test_set, param_indices, E):
    print "Organizing data.."
    # organizes data as a matrix for linear regression    
    hr_ind = param_indices["hr"]
    W = get_workout_count(train_set)
    assert(W == get_workout_count(val_set) and (test_set is None or W == get_workout_count(test_set)))
    
    wins = np.array(range(0, W))
    w_indices_train = list(np.searchsorted(extract_col(train_set, 0), wins))
    w_indices_train.append(train_set.shape[0])
    w_indices_val = list(np.searchsorted(extract_col(val_set, 0), wins))
    w_indices_val.append(val_set.shape[0])
    if (test_set is not None):
        w_indices_test = list(np.searchsorted(extract_col(test_set, 0), wins))
        w_indices_test.append(test_set.shape[0])

    train_X = [0.0] * W
    val_X = [0.0] * W
    test_X = [0.0] * W
    train_y = [0.0] * W
    val_y = [0.0] * W
    test_y = [0.0] * W

    for w in xrange(0, W):
        # training
        start, end = w_indices_train[w : w + 2]
        hr_train = train_set[start:end, hr_ind]
        X, y = organize_for_learning(hr_train, E)
        train_X[w] = X
        train_y[w] = y

        # validation
        start, end = w_indices_val[w : w + 2]
        hr_val = val_set[start:end, hr_ind]
        hr_val = np.concatenate((hr_train[-E:], hr_val))
        X, y = organize_for_learning(hr_val, E)
        val_X[w] = X
        val_y[w] = y

        # test
        if (test_set is not None):
            start, end = w_indices_test[w : w + 2]
            hr_test = test_set[start:end, hr_ind]
            hr_test = hr_val[-E:] + hr_test
            X,y = organize_for_learning(hr_test, E)
            test_X[w] = X
            test_y[w] = y
        
        if (w % 10000 == 0):
            print "Done %d out of %d" % (w, W)
   
    return train_X, train_y, val_X, val_y, test_X, test_y
    
def learn(all_X, all_y, E):
    W = len(all_X) 
    models = [0.0] * W
    for w in xrange(0, W):
        X = all_X[w]
        y = all_y[w]
        #model = linear_model.LinearRegression(copy_X = True)
        model = linear_model.Ridge(alpha = 100000000.0, copy_X = True)
        model.fit(X, y)
        models[w] = model
        
        #arma = sm.tsa.ARMA(y, (E, 1))
        #model = sm.tsa.AR(y)
        #models[w] = model.fit(maxlag = E)
        #models[w] = arma.fit(start_params = np.zeros((E+2, 1)))
        if (w % 10000 == 0):
            print "Done %d workouts out of %d" % (w, W)
    return models
    
def get_samples_per_workout(data):
    assert(type(data).__name__ == "matrix" or type(data).__name__ == "ndarray")
    N = data.shape[0]
    U = get_workout_count(data)
    uins = np.array(range(0, U))
    if (type(data).__name__ == "matrix"):
        col0 = data[:, 0].A1
    else:
        col0 = data[:, 0]
    u_indices = list(np.searchsorted(col0, uins))
    u_indices.append(N)
    workouts_per_user = [0] * U
    for i in range(0, U):
        start_u = u_indices[i]
        end_u = u_indices[i+1]
        workouts_per_user[i] = end_u - start_u
    return workouts_per_user

def check_sorted(data, param_indices):
    N = data.shape[0]
    ind_u = param_indices["workout_number"]
    ind_t = param_indices["duration"]
    i = 0
    while (i < N):
        u = (data[i, ind_u])
        i += 1
        while i < N and data[i, ind_u] == u:
            assert(data[i, ind_t] >= data[i - 1, ind_t])
            i += 1
        if (i < N):
            assert(data[i - 1, ind_u] < data[i, ind_u])

def flatten_list(l):
    new_l = [x for sublist in l for x in sublist]
    return new_l

def main():
    t1 = time.time()

    E = int(sys.argv[1])
 
    #infile = "../../data/endoMondo5000_workouts.gz"
    infile = "../../data/all_workouts.gz"
    mode = "final"  # can be "final" or "random"
    
    outfile = infile + mode + "_inst_many.npz"

    # prepare data set.. Run once and comment it out if running multiple times with same settings
    #prepare(infile, outfile, mode)

    print "Loading data from file.."
    data = np.load(outfile)
    train_set = data["train_set"]
    val_set = data["val_set"]
    test_set = data["test_set"]
    param_indices = data["param_indices"][()]
    #print "Doing sorted check on train and val sets.."
    #check_sorted(train_set, param_indices)
    #check_sorted(val_set, param_indices)

    print "Number of workouts = ", get_workout_count(train_set)
    print "Training set has %d examples" % (train_set.shape[0])
    print "Validation set has %d examples" % (val_set.shape[0])

    #train_X, train_y, val_X, val_y, test_X, test_y = organize_data(train_set, val_set, None, param_indices, E)
    train_X, train_y = organize(train_set, E, param_indices)
    #train_X = train_X[:100]
    #train_y = train_y[:100]

    print "Training.."
    all_models = learn(train_X, train_y, E)
    #np.savez("model.npz", theta = theta, sigma = sigma, E = E)
    
    #print "Loading model.."
    #model = np.load("model.npz")
    #theta = model["theta"]

    # add the experience level to each workout in train, validation and test set
    #print "Adding experience levels to data matrices"
    #train_set = add_experience_column_to_train_set(train_set, sigma_train, param_indices)
    #val_set = add_experience_column_to_train_set(val_set, sigma_val, param_indices)
    #val_set = add_experience_column_to_test_set(val_set, train_set, param_indices, mode = mode)

    print "Making predictions (train).."
    W = len(train_X)
    all_last_E = [0.0] * W
    for w in xrange(0, W):
        all_last_E[w] = train_y[w][:E]
    next_n = get_samples_per_workout(train_set)
    for w in range(0, W):
        next_n[w] = next_n[w] - E
    train_pred = predict_next(all_last_E, all_models, next_n, E)
    
    print "Making predictions (val).. "
    for w in xrange(0, W):
        all_last_E[w] = train_y[w][-E:]
        assert(len(all_last_E[w]) == E)
    val_pred = predict_next(all_last_E, all_models, get_samples_per_workout(val_set), E)

    print "Makign predictions (test) .."
    for w in xrange(0, W):
        all_last_E[w] = val_pred[w][-E:]
        assert(len(all_last_E[w]) == E)
    test_pred = predict_next(all_last_E, all_models, get_samples_per_workout(test_set), E)

    #with open("pred.txt", "w") as f:
    #    f.write(str(train_pred[-200:]))
    #with open("true.txt", "w") as f:
    #    f.write(str(train_y[-200:]))
    #with open("diff.txt", "w") as f:
    #    f.write(str(list(train_y.A1 - train_pred.A1)))
    
    train_pred = np.matrix(flatten_list(train_pred)).T
    val_pred = np.matrix(flatten_list(val_pred)).T
    test_pred = np.matrix(flatten_list(test_pred)).T
    
    train_y = np.matrix(flatten_list(train_y)).T
    val_y = val_set[:, param_indices["hr"]]
    val_y = np.matrix(list(val_y)).T
    test_y = test_set[:, param_indices["hr"]]
    test_y = np.matrix(list(test_y)).T

    n = train_pred.shape[0]
    print n
    for i in xrange(0, n):
        d = train_pred[i, 0] - train_y[i, 0]
        if (d > 200):
            print "ALERT : ", i
            print train_pred[i, 0]
            print train_y[i, 0]
            break


    print "Computing statistics"
    [mse, var, fvu, r2, errors] = compute_stats(train_y, train_pred)
    print "\n@Training Examples = %d,MSE = %f,Variance = %f,FVU = %f,R2 = 1 - FVU = %f, E = %d\n" % (train_set.shape[0],mse, var, fvu, r2, E)
    [mse, var, fvu, r2, errors] = compute_stats(val_y, val_pred)
    print "@Validation Examples = %d,MSE = %f,Variance = %f,FVU = %f,R2 = 1 - FVU = %f, E = %d\n" % (val_set.shape[0],mse, var, fvu, r2, E)
    [mse, var, fvu, r2, errors] = compute_stats(test_y, test_pred)
    print "@Test Examples = %d,MSE = %f,Variance = %f,FVU = %f,R2 = 1 - FVU = %f, E = %d\n" % (test_set.shape[0],mse, var, fvu, r2, E)

    t2 = time.time()
    print "@Total time taken = ", t2 - t1

    #plt.figure()
    #plt.subplot(1,2,0)
    #plot_data(train_set, train_pred, param_indices, title = "Training set")
    #plt.subplot(1,2,1)
    #plot_data(val_set, val_pred, param_indices, title = "Validation set")

    #plt.show()


if __name__ == "__main__":
    main()
