import gzip
import utils
import numpy as np
import math

def read_data(infile, x_params, y_param):
    X = []
    Y = []
    n = 0
    with gzip.open(infile) as f:
        for line in f:
            d = utils.json_to_dict(line)
            ignore = False
            xrow = [1]
            for xp in x_params:
                if (not d.has_key(xp)):
                    ignore = True
                    break
                xrow.append(d[xp])
            if (not d.has_key(y_param)):
                ignore = True
            else:
                y = d[y_param]
            if (not ignore):
                assert(len(xrow) == len(x_params) + 1)
                X.append(xrow)
                Y.append(y)
            n += 1
            if (n % 100000 == 0):
                print "%d workouts read.." % (n)

    assert(len(X) == len(Y))
    X = np.matrix(X)
    Y = np.matrix(Y).T
    return [X,Y]

def shuffle_examples(X, Y):
    # randomize examples (shuffle X and Y in unison)
    ncols = X.shape[1]
    XY = np.concatenate((X, Y), axis=1)
    np.random.shuffle(XY)
    X = XY[:, 0 : ncols]
    Y = XY[:, ncols : ]
    return [X, Y]

def split_examples(X, Y):
    # split data into train, validation and test sets
    train_percent = 0.10
    val_percent = 0.10
    test_percent = 1.0 - train_percent - val_percent
    N = X.shape[0]

    train_end = math.ceil(float(N) * train_percent)
    X_train = X[0 : train_end, :]
    Y_train = Y[0 : train_end, :]
    
    val_end = train_end + (float(N) * val_percent)
    X_val = X[train_end : val_end, :]
    Y_val = Y[train_end : val_end, :]

    X_test = X[val_end : , :]
    Y_test = Y[val_end : , :]

    assert(X_train.shape[0] == Y_train.shape[0])
    assert(X_val.shape[0] == Y_val.shape[0])
    assert(X_test.shape[0] == Y_test.shape[0])

    print "Training set has " + str(X_train.shape[0]) + " examples"
    print "Validation set has " + str(X_val.shape[0]) + " examples"
    print "Test set has " + str(X_test.shape[0]) + " examples"

    return [X_train, Y_train, X_val, Y_val, X_test, Y_test]

def prepare_train_val_test_sets(infile, x_params, y_param, outfile_base=""):
    [X, Y] = read_data(infile, x_params, y_param)    # read data
    [X, Y] = shuffle_examples(X, Y)  # shuffle examples
    [X_train, Y_train, X_val, Y_val, X_test, Y_test] = split_examples(X, Y)  # split
    np.save(outfile_base + "X_train.npy", X_train)
    np.save(outfile_base + "Y_train.npy", Y_train)
    np.save(outfile_base + "X_val.npy", X_val)
    np.save(outfile_base + "Y_val.npy", Y_val)
    np.save(outfile_base + "X_test.npy", X_test)
    np.save(outfile_base + "Y_test.npy", Y_test)

if __name__ == "__main__":
    prepare_train_val_test_sets("endoMondo5000_workouts_condensed.gz", ["Distance", "pace(avg)", "hr(avg)","alt(avg)"],"Duration")
