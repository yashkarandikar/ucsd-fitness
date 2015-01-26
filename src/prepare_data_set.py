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
        return "Feature : %s absent in data" % (self.feature)

def read_data_as_lists(infile, sport, params):
    print "Infile : ", infile
    print "params : ", params
    sport_missing = 0
    param_missing = 0
    n_ignore = 0
    n = 0
    data = []
    with gzip.open(infile) as f:
        for line in f:
            d = utils.json_to_dict(line)
            example = []
            ignore = False
            if (d["sport"] != sport):
                ignore = True
                sport_missing += 1
            else:
                for k in params:
                    if not d.has_key(k):
                        param_missing += 1
                        ignore = True
                        break
                    else:
                        example.append(d[k])
            if (ignore):
                n_ignore += 1
            else:
                data.append(example)
            n += 1
            if (n % 100000 == 0):
                print "%d workouts read.." % (n)

    print "%d workouts did not match the sport" % (sport_missing)
    print "%d workouts did not contain one or more parameters" % (param_missing)
    print "%d workouts ignored.." % (n_ignore)
    print "%d workouts successfully returned.." % (len(data))
    return data


def read_data(infile, sport, x_params, y_param, missing_data_mode = "substitute"):
    # returns matrices X and y (a single column). An extra binary feature is is introduced for each feature to indicate presence/absence of data. MIssing values are put as 0.
    X = []
    y = []
    n = 0
    n_ignore = 0
    x_missing = 0
    y_missing = 0
    sport_missing = 0
    print "Infile : ", infile
    print "X params : ", x_params
    print "y param : ", y_param
    with gzip.open(infile) as f:
        for line in f:
            ignore = False
            d = utils.json_to_dict(line)
            if (d["sport"] != sport):
                ignore = True
                sport_missing += 1
            else:
                xrow = []
                if (missing_data_mode == "ignore"):
                    for xp in x_params:
                        if (not d.has_key(xp)):
                            ignore = True
                            x_missing += 1
                            break
                        xrow.append(d[xp])
                    if (not d.has_key(y_param)):
                        ignore = True
                        y_missing += 1
                    else:
                        y_val = d[y_param]
                    if (not ignore):
                        assert(len(xrow) == len(x_params))
                        X.append(xrow)
                        y.append(y_val)
                elif ((not ignore) and missing_data_mode == "substitute"):
                    # if data is missing, add a feature [0] and put value as 0 for now, else add a feature [1] and put value as actual value
                    x_param_missing = False
                    for xp in x_params:
                        if (not d.has_key(xp)):
                            xrow.append(0)  # binary feature
                            xrow.append(0.0)    # missing value
                            x_param_missing = True
                        else:
                            xrow.append(1)  # binary feature
                            xrow.append(d[xp]) # value
                    if (x_param_missing):
                        x_missing += 1
                    if (not d.has_key(y_param)):
                        ignore = True
                        y_missing += 1
                    else:
                        y_val = d[y_param]
                    if (not ignore):    
                        # consider example only if y value is present
                        assert(len(xrow) == 2 * len(x_params))
                        y.append(d[y_param])
                        X.append(xrow)
                else:
                    raise Exception("invalid missing_data_mode")

            if (ignore):
                n_ignore += 1

            n += 1
            if (n % 100000 == 0):
                print "%d workouts read.." % (n)

    assert(len(X) == len(y))
    X = np.matrix(X)
    y = np.matrix(y).T
    print "%d workouts did not match the sport" % (sport_missing)
    print "%d workouts did not contain one or more X parameters" % (x_missing)
    print "%d workouts did not contain Y parameter" % (y_missing)
    print "%d workouts ignored.." % (n_ignore)
    return [X,y]


def handle_missing_data(X, x_params):
    # assumes columns are in the form: presence/absence feature followed by feature value, then 2nd feature, so on
    print "Handling missing data.."
    
    if (X.shape[0] == 0 or X.shape[1] == 0):
        return
    
    ncols = X.shape[1]
    assert(len(x_params) == ncols / 2)
    
    # compute means of columns
    col_means = np.sum(X, axis = 0)
    j = 1
    while j < ncols:
        if (col_means[0, j - 1] > 0):
            col_means[0, j] /= col_means[0, j - 1]
            j += 2
        else:
            raise FeatureAbsentException(x_params[j/2])

    # replace missing values by mean
    nrows = X.shape[0]
    for i in range(0, nrows):
        j = 0
        while (j < ncols - 1):
            if (X[i, j] == 0):  # if missing
                X[i, j+1] = col_means[0, j + 1]
            j += 2

def normalize_data(X, missing_data_mode = "substitute"):
    print "Normalizing.."
    ncols = X.shape[1]
    means = np.mean(X, axis = 0)
    stds = np.std(X, axis = 0)
    if (missing_data_mode == "substitute"):
        for i in range(0, ncols):
            if i % 2 == 0:
                means[0, i] = 0.0
                stds[0, i] = 1.0
    X = (X - means) / stds;
    return X

def add_offset_feature(X):
    print "Adding offset feature.."
    nrows = X.shape[0]
    ones = np.ones((nrows, 1))
    return np.concatenate((ones, X), axis = 1)

def generate_param_indices(x_params, missing_data_mode, with_intercept):
    d = {}
    assert(missing_data_mode == "ignore" or missing_data_mode == "substitute")
    ind = 0
    if (with_intercept):
        d["intercept"] = 0
        ind += 1
    if (missing_data_mode == "ignore"):
        for i in range(0, len(x_params)):
            d[x_params[i]] = ind; ind += 1
        assert(len(d.keys()) == len(x_params) + 1.0 * with_intercept)
    else:
        for i in range(0, len(x_params)):
            d[x_params[i] + "_present"] = ind; ind += 1
            d[x_params[i]] = ind; ind += 1
        assert(len(d.keys()) == (2 * len(x_params)) + 1.0 * with_intercept)
    return d

def prepare_data_set(infile, sport, x_params, y_param, outfile, split_fraction, shuffle_before_split = True, missing_data_mode = "substitute", normalize = True, outlier_remover = None):
    try:
        randomState = np.random.RandomState(seed = 12345)
        print "X params = " + str(x_params)
        print "Y param = " + str(y_param)
        print "Options : missing_data_mode = " + missing_data_mode +  ", normalize = " + str(normalize) + " split fraction = " + str(split_fraction)
        print "Reading data.."
        [X, y] = read_data(infile, sport, x_params, y_param, missing_data_mode = missing_data_mode)    # read data
        param_indices = generate_param_indices(x_params, missing_data_mode, with_intercept = False)

        # remove outliers
        if (outlier_remover is not None):
            [X, y] = outlier_remover(X, y, x_params, y_param, missing_data_mode, param_indices)

        # Shuffle and split data
        [X1, y1, X2, y2] = utils.shuffle_and_split_Xy(X, y, split_fraction, randomState = randomState, shuffle = shuffle_before_split)

        # substitute for missing values
        if (missing_data_mode == "substitute"):
            handle_missing_data(X1, x_params)
            handle_missing_data(X2, x_params)
        
        # Normalization for unit variance and zero mean
        if (normalize):
            X1 = normalize_data(X1, missing_data_mode = missing_data_mode)
            X2 = normalize_data(X2, missing_data_mode = missing_data_mode)

        # add an intercept feature of all 1's
        X1 = add_offset_feature(X1)
        X2 = add_offset_feature(X2)
        param_indices = generate_param_indices(x_params, missing_data_mode, with_intercept = True)

        print "Writing to disk.."
        np.savez(outfile, X1 = X1, y1 = y1, X2 = X2, y2 = y2, param_indices = param_indices)
    except FeatureAbsentException as e:
        print e
        print "No output files written.."
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reads .gz file, extracts given X params, given y param and saves as a single .npz files')
    parser.add_argument('--infile', type=str, help='.gz file', dest='infile')
    parser.add_argument('--x-params', type=str, help='comma separated list of X parameters', dest='x_param_list')
    parser.add_argument('--y-param', type=str, help='y param', dest='y_param')
    parser.add_argument('--outfile', type=str, help='output .npz file', dest='outfile', default="")
    args = parser.parse_args()
    if (args.infile is None or args.x_param_list is None or args.y_param is None):
        parser.print_usage()
        sys.exit(0)
    else:
        infile = args.infile
        y_param = args.y_param
        x_params = args.x_param_list.split(",")
        x_params = [s.strip() for s in x_params]
        prepare_data_set(infile=infile, x_params = x_params, y_param = y_param, outfile = args.outfile, missing_data_mode="ignore", normalize=True)
