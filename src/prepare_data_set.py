import gzip
import utils
import numpy as np
import math
import argparse
import sys

class FeatureAbsentException(Exception):
    def __init__(self, feature):
        self.feature = feature

    def __str__(self):
        return "Feature : %s" % (self.feature)


def read_data(infile, x_params, y_param):
    # returns matrices X and y (a single column). An extra binary feature is is introduced for each feature to indicate presence/absence of data. MIssing values are put as 0. These should be replaced by mean AFTER splitting into train, test and validation sets
    X = []
    y = []
    n = 0
    n_ignore = 0
    print "Infile : ", infile
    print "X params : ", x_params
    print "y param : ", y_param
    with gzip.open(infile) as f:
        for line in f:
            d = utils.json_to_dict(line)
            ignore = False
            xrow = [1]
            """
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
                y.append(y)
            """
            # if data is missing, add a feature [0] and put value as 0 for now, else add a feature [1] and put value as actual value
            for xp in x_params:
                if (not d.has_key(xp)):
                    xrow.append(0)  # binary feature
                    xrow.append(0.0)    # missing value
                else:
                    xrow.append(1)  # binary feature
                    xrow.append(d[xp]) # value
            if (d.has_key(y_param)):    
                # consider example only if y value is present
                y.append(d[y_param])
                X.append(xrow)
            else:
                n_ignore += 1

            n += 1
            if (n % 100000 == 0):
                print "%d workouts read.." % (n)

    assert(len(X) == len(y))
    X = np.matrix(X)
    y = np.matrix(y).T
    print "%d workouts ignored.." % (n_ignore)
    return [X,y]


def handle_missing_data(X, x_params):
    # assumes columns are in the form: 1st column is constant 1, then presence/absence feature followed by feature value, then 2nd feature, so on
    ncols = X.shape[1]
    assert(len(x_params) == (ncols - 1) / 2)
    
    # compute means of columns
    col_means = np.sum(X, axis = 0)
    j = 2
    while j < ncols:
        if (col_means[0, j - 1] > 0):
            col_means[0, j] /= col_means[0, j - 1]
            j += 2
        else:
            raise FeatureAbsentException(x_params[j/2 - 1])

    # replace missing values by mean
    nrows = X.shape[0]
    for i in range(0, nrows):
        j = 1
        while (j < ncols - 1):
            if (X[i, j] == 0):  # if missing
                X[i, j+1] = col_means[0, j + 1]
            j += 2

def prepare_data_set(infile, x_params, y_param, outfile_base=""):
    try:
        print "Reading data.."
        [X, y] = read_data(infile, x_params, y_param)    # read data
        print "Substituting missing data by means"
        handle_missing_data(X, x_params)
        print "Writing npy files.."
        np.save(outfile_base + "X.npy", X)
        np.save(outfile_base + "y.npy", y)
    except FeatureAbsentException as e:
        print e
        print "No output files written.."
    #[X, y] = shuffle_examples(X, y)  # shuffle examples
    #[X_train, y_train, X_val, y_val, X_test, y_test] = split_examples(X, y)  # split
    #handle_missing_data(X_train)
    #handle_missing_data(X_val)
    #handle_missing_data(X_train)
    #np.save(outfile_base + "X_train.npy", X_train)
    #np.save(outfile_base + "y_train.npy", y_train)
    #np.save(outfile_base + "X_val.npy", X_val)
    #np.save(outfile_base + "y_val.npy", y_val)
    #np.save(outfile_base + "X_test.npy", X_test)
    #np.save(outfile_base + "y_test.npy", y_test)

if __name__ == "__main__":
    #prepare_train_val_test_sets("endoMondo5000_workouts_condensed.gz", ["Distance", "pace(avg)", "hr(avg)","alt(avg)"],"Duration","5000_")
    #prepare_train_val_test_sets("../../data/all_workouts_condensed.gz", ["Distance", "pace(avg)", "hr(avg)","alt(avg)"],"Duration","full_")
    parser = argparse.ArgumentParser(description='Reads .gz file, extracts given X params, given y param and saves X and y .npy files')
    parser.add_argument('--infile', type=str, help='.gz file', dest='infile')
    parser.add_argument('--x-params', type=str, help='comma separated list of X parameters', dest='x_param_list')
    parser.add_argument('--y-param', type=str, help='y param', dest='y_param')
    parser.add_argument('--outfile-base', type=str, help='prefix name of output .npy files', dest='outfile_base', default="")
    args = parser.parse_args()
    if (args.infile is None or args.x_param_list is None or args.y_param is None):
        parser.print_usage()
        sys.exit(0)
    else:
        infile = args.infile
        y_param = args.y_param
        x_params = args.x_param_list.split(",")
        x_params = [s.strip() for s in x_params]
        prepare_data_set(infile=infile, x_params = x_params, y_param = y_param, outfile_base = args.outfile_base)

"""
def shuffle_examples(X, y):
    # randomize examples (shuffle X and y in unison)
    ncols = X.shape[1]
    Xy = np.concatenate((X, y), axis=1)
    np.random.shuffle(Xy)
    X = Xy[:, 0 : ncols]
    y = Xy[:, ncols : ]
    return [X, y]

def split_into_2(X, y, fraction):
    # split data into 2
    N = X.shape[0]
    end1 = math.ceil(float(N) * train_percent)
    X_1 = X[0 : end1, :]
    y_1 = y[0 : end1, :]
    
    X_2 = X[end1 : , :]
    y_2 = y[end1 : , :]

    assert(X_1.shape[0] == y_1.shape[0])
    assert(X_2.shape[0] == y_2.shape[0])

    print "Set 1 has " + str(X_1.shape[0]) + " examples"
    print "Set 2 has " + str(X_val.shape[0]) + " examples"

    return [X_1, y_1, X_2, y_2]


def split_examples_train_test_val(X, y):
    # split data into training test val sets
    train_percent = 0.10
    val_percent = 0.10
    test_percent = 1.0 - train_percent - val_percent

    N = X.shape[0]
    train_end = math.ceil(float(N) * train_percent)
    X_train = X[0 : train_end, :]
    y_train = y[0 : train_end, :]
    
    val_end = train_end + (float(N) * val_percent)
    X_val = X[train_end : val_end, :]
    y_val = y[train_end : val_end, :]

    X_test = X[val_end : , :]
    y_test = y[val_end : , :]

    assert(X_train.shape[0] == y_train.shape[0])
    assert(X_val.shape[0] == y_val.shape[0])
    assert(X_test.shape[0] == y_test.shape[0])

    print "Training set has " + str(X_train.shape[0]) + " examples"
    print "Validation set has " + str(X_val.shape[0]) + " examples"
    print "Test set has " + str(X_test.shape[0]) + " examples"

    return [X_train, y_train, X_val, y_val, X_test, y_test]
"""
