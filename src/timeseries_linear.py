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

import pyximport; pyximport.install()
from timeseries_linear_pyx import F_pyx

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
        x = [0.0] * int(n + E)
        for i in xrange(0, E):
            x[i] = all_last_E[w][i]
        for i in xrange(0, n):
            #p = m.predict(np.matrix([x]))[0]
            #p = m.predict(np.insert(v, 0, 1.0))
            v = x[i:i+E]
            v_temp = np.insert(v, 0, 1.0)
            p = np.dot(m, v_temp)
            all_y[w][i] = p
            try :
                x[E + i] = p
            except:
                print "==="
                print E
                print i
                print len(x)
                print next_n[w]
                print n
                print "==="
        if (w % 10000 == 0):
            print "Done %d workouts, out of %d" % (w, W)

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

def linreg(X, y, lam):
    #theta,residuals,rank,s = np.linalg.lstsq(X, y)
    randomState = np.random.RandomState(12345)
    M = X.shape[1]
    check_grad = False
    theta = randomState.rand(M)
    F_fn = F_pyx
    
    if (check_grad):
        #print F(theta, X, y, lam)[1]
        our_grad = np.linalg.norm(F_fn(theta, X, y, lam)[1], ord = 2)
        numerical = np.linalg.norm(scipy.optimize.approx_fprime(theta, F_fn, np.sqrt(np.finfo(np.float).eps), X, y, lam, True), ord = 2)
        print "our grad = ", our_grad
        print "numerical = ", numerical
        ratio = our_grad / numerical
        print "Check grad ratio = ", ratio
        assert(abs(1.0 - ratio) < 1e-4)

    [theta, E_min, info] = scipy.optimize.fmin_l_bfgs_b(F_fn, theta, None, args = (X, y, lam))
    
    return theta


def learn(all_X, all_y, lam, E):
    W = len(all_X) 
    models = [0.0] * W
    for w in xrange(0, W):
        X = all_X[w]
        y = all_y[w]
        #model = linear_model.LinearRegression(copy_X = True)
        #model = linear_model.Ridge(alpha = lam, copy_X = True, normalize = True)
        #model.fit(X, y)
        #models[w] = model
        theta = linreg(X, y, lam)
        models[w] = theta
        
        #arma = sm.tsa.ARMA(y, (E, 1))
        #model = sm.tsa.AR(y)
        #models[w] = model.fit(maxlag = E)
        #models[w] = arma.fit(start_params = np.zeros((E+2, 1)))
        if (w % 1000 == 0):
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

def find_most_common_interval(a):
    n = a.shape[0]
    diffs = {}
    for i in xrange(0, n - 1):
        d = a[i+1] - a[i]
        if (not diffs.has_key(d)):
            diffs[d] = 0
        diffs[d] += 1
    max_v = 0
    for k, v in diffs.items():
        if (v > max_v):
            max_k = k
            max_v = v
    return max_k

def sort_matrix_by_2_cols(m, col1, col2):
    key_cols = (m[:, col2], m[:, col1])
    return m[np.lexsort(key_cols)]

def correct_missing(data, param_indices):
    ind_dur = param_indices["duration"]
    ind_hr = param_indices["hr"]
    N = data.shape[0]
    W = get_workout_count(data)
    samples_per_workout = get_samples_per_workout(data)
    n_odd_interval_workouts = 0
    n_missing_sample_workouts = 0
    n_ignored = 0
    new_rows = []
    i = 0
    for w in xrange(0, W):
        nw = samples_per_workout[w]
        interval = int(find_most_common_interval(data[i:i+nw, ind_dur]))
        #if (w == 33205):
        #    print "Workout ", w
        #    print "Number of samples = ", nw
        #    print "interval = ", interval
        #    print list(data[i:i+nw, ind_dur])
        #    print list(data[i:i+nw, ind_hr])
        #i += nw
        #continue
        missing_samples = False
        odd_intervals = False
        ignored = False
        for j in xrange(0, nw):
            if (j < nw - 1):
                hr_i = data[i, ind_hr]
                hr_ipp = data[i+1, ind_hr]
                dur_i = data[i, ind_dur]
                dur_ipp = data[i+1, ind_dur]
                d_dur = int(dur_ipp - dur_i)
                if (d_dur > interval):
                    # add intermediate points
                    #assert(d_dur % interval == 0)
                    if (d_dur % interval == 0):
                        n_new = (d_dur / interval) - 1
                        if (n_new <= 10000):
                            t = dur_i
                            hr = hr_i
                            t_step = interval
                            hr_step = (hr_ipp - hr_i) / (n_new + 1)
                            while t < data[i+1, ind_dur]:
                                #print "workout = %d, prev = %d, t = %d, next = %d"  % (w, dur_i, t, dur_ipp)
                                t += t_step
                                hr += hr_step
                                new_row = list(data[i, :])
                                new_row[ind_dur] = t
                                new_row[ind_hr] = hr
                                new_rows.append(new_row)
                                mem_used = ps.phymem_usage().percent
                                if (mem_used > 70):
                                    print "Too much memory being used.."
                                    sys.exit(0)
                        else:
                            ignored = True
                        missing_samples = True
                    elif (d_dur % interval != 0):
                        odd_intervals = True
                        ignored = True
            i += 1
            mem_used = ps.phymem_usage().percent
            if (mem_used > 70):
                print "Too much memory being used.."
                sys.exit(0)
        if (missing_samples):
            n_missing_sample_workouts += 1
        if (odd_intervals):
            n_odd_interval_workouts += 1
        if (ignored):
            n_ignored += 1
            print "Ignored.."
        if (w % 10000 == 0):
            print "Done with %d .." % (w)
    assert(i == N)

    data = np.concatenate((data, new_rows), axis = 0)
    data = sort_matrix_by_2_cols(data, param_indices["workout_id"], param_indices["duration"])
    print "# workouts with missing samples = ", n_missing_sample_workouts
    print "# workouts with odd intervals = " , n_odd_interval_workouts
    print "# workouts not corrected = ", n_ignored
    return data
    
def main():
    t1 = time.time()

    lam = float(sys.argv[1])
    E = int(sys.argv[2])
 
    #infile = "../../data/endoMondo5000_workouts.gz"
    infile = "../../data/all_workouts.gz"
    mode = "final"  # can be "final" or "random"
    
    outfile = infile + mode + "_inst_many.npz"
    outfile2 = outfile + ".corrected.npz"

    # prepare data set.. Run once and comment it out if running multiple times with same settings
    #prepare(infile, outfile, mode)

    print "Loading data from file.."
    data = np.load(outfile)
    train_set = data["train_set"]
    val_set = data["val_set"]
    try:
        test_set = data["test_set"]
    except KeyError:
        test_set = None
    param_indices = data["param_indices"][()]

    print "Number of workouts = ", get_workout_count(train_set)
    print "Training set has %d examples" % (train_set.shape[0])
    print "Validation set has %d examples" % (val_set.shape[0])
    
    # correct for missing values
    print "Correcting missing values.."
    train_set = correct_missing(train_set, param_indices)
    val_set = correct_missing(val_set, param_indices)
    test_set = correct_missing(test_set, param_indices)
    np.savez(outfile2, train_set = train_set, val_set = val_set, test_set = test_set, param_indices = param_indices)

    print "Loading corrected data.."
    data = np.load(outfile2)
    train_set = data["train_set"]
    val_set = data["val_set"]
    test_set = data["test_set"]
    param_indices = data["param_indices"][()]
    print "Training set : ", train_set.shape
    print "Validation set : ", val_set.shape
    print "Testing set : ", test_set.shape

    # process data
    train_X, train_y = organize(train_set, E, param_indices)
    W = len(train_X)
    for w in xrange(0, W):
        assert(train_X[w].shape[1] == E + 1)

    # training
    print "Training.."
    all_models = learn(train_X, train_y, lam, E)
    
    print "Making predictions (train).."
    W = len(train_X)
    all_last_E = [0.0] * W
    for w in xrange(0, W):
        all_last_E[w] = train_X[w][0, 1:E+1]
        assert(len(all_last_E[w]) == E)
    next_n = get_samples_per_workout(train_set)
    for w in range(0, W):
        next_n[w] = next_n[w] - E
    train_pred = predict_next(all_last_E, all_models, next_n, E)
    
    print "Making predictions (val).. "
    for w in xrange(0, W):
        #all_last_E[w] = train_y[w][-E:]
        l = train_X[w][-1, 2:E+1]
        all_last_E[w] = np.insert(l, E - 1, train_y[w][-1])
        assert(len(all_last_E[w]) == E)
    val_pred = predict_next(all_last_E, all_models, get_samples_per_workout(val_set), E)

    if (test_set is not None):
        print "Making predictions (test) .."
        for w in xrange(0, W):
            all_last_E[w] = val_pred[w][-E:]
            assert(len(all_last_E[w]) == E)
        test_pred = predict_next(all_last_E, all_models, get_samples_per_workout(test_set), E)

    train_pred = np.matrix(flatten_list(train_pred)).T
    val_pred = np.matrix(flatten_list(val_pred)).T
    if (test_set is not None):
        test_pred = np.matrix(flatten_list(test_pred)).T
    
    train_y = np.matrix(flatten_list(train_y)).T
    val_y = val_set[:, param_indices["hr"]]
    val_y = np.matrix(list(val_y)).T
    if (test_set is not None):
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

    print "Computing statistics"
    [mse, var, fvu, r2, errors] = compute_stats(train_y, train_pred)
    print "\n@Training Examples = %d,MSE = %f,Variance = %f,FVU = %f,R2 = 1 - FVU = %f, E = %d\n" % (train_set.shape[0],mse, var, fvu, r2, E)
    print val_y.shape
    print val_pred.shape
    [mse, var, fvu, r2, errors] = compute_stats(val_y, val_pred)
    print "@Validation Examples = %d,MSE = %f,Variance = %f,FVU = %f,R2 = 1 - FVU = %f, E = %d\n" % (val_set.shape[0],mse, var, fvu, r2, E)
    if (test_set is not None):
        [mse, var, fvu, r2, errors] = compute_stats(test_y, test_pred)
        print "@Test Examples = %d,MSE = %f,Variance = %f,FVU = %f,R2 = 1 - FVU = %f, E = %d\n" % (test_set.shape[0],mse, var, fvu, r2, E)

    t2 = time.time()
    print "@Total time taken = ", t2 - t1

if __name__ == "__main__":
    main()
