import simplejson as json
import os
import gzip
import numpy as np
import math
import random

def get_workouts(infile):
    # assumes each line of the input file is a json structure
    dicts = []
    fName, fExt = os.path.splitext(infile) 
    f = 0
    if (fExt == ".gz"):
        f = gzip.open(infile)
    elif (fExt == ".txt"):
        f = open(infile)
    else:
        print fExt
        raise Exception("Invalid file format")
    
    for line in f:
        dicts.append(json.loads(line))
    f.close()
    return dicts

def json_to_dict(s):
    # s is a json formatted string
    return json.loads(s)

def dict_to_json(d):
    # d is a dictionary
    return json.dumps(d)

def remove_null_values_single(l):
    # given list of values, removes those marked 'N'
    return [x for x in l if x != 'N']

def remove_null_values(l1, l2):
    # given list of values, removes those marked 'N'
    assert(len(l1) == len(l2))
    n = len(l1)
    l1_new = []
    l2_new = []
    for i in range(0, n):
        if (l1[i] != 'N' and l2[i] != 'N'):
            l1_new.append(l1[i])
            l2_new.append(l2[i])
    return [l1_new, l2_new]

def get_user_id_from_filename(infile):
    f = os.path.basename(infile)
    parts = f.split(".")
    if (len(parts) != 2):
        raise Exception("Filename is not in recognized format")
    return int(parts[0])

def combine_gzip_files(files, outfile):
    # combines multiple gzip files into one single gzipped file
    command = "cat"
    for f in files:
        command = command + " " + f
    command = command + " > " + outfile
    os.system(command)

def append_to_base_filename(infile, s):
    """
    if infile is of the form name.ext, then this function will return (name+s).ext
    """
    fName, fExt = os.path.splitext(infile)
    return fName + s + fExt

def combine_Xy(X, y):
    return np.concatenate((X, y), axis = 1)

def separate_Xy(Xy):
    X = Xy[:,:-1]
    y = Xy[:,-1:]
    return [X, y]

def sort_matrix_by_col(m, i):
    return m[np.array(m[:,i].argsort(axis=0).tolist()).ravel()]
    #return m[m[:,i].argsort()]

def sort_data_by_col(X, y, i):
    Xy = combine_Xy(X, y)
    Xy = sort_matrix_by_col(Xy, i)
    [X, y] = separate_Xy(Xy)
    return [X, y]

def remove_rows_by_condition(m, cols, lower_bounds, upper_bounds):
    assert(len(cols) == len(lower_bounds) and len(lower_bounds) == len(upper_bounds))
    n = len(cols)
    for i in range(0, n):
        c = cols[i]
        m = sort_matrix_by_col(m, c)
        l = lower_bounds[i]; u = upper_bounds[i]
        i = np.searchsorted(m[:, c].A1, l)
        j = np.searchsorted(m[:, c].A1, u)
        m = m[i:j, :]
    return m

def shuffle_and_split_Xy(X, y, fraction, randomState = None, shuffle = True):
    Xy = combine_Xy(X, y)
    if (shuffle):
        if (randomState is not None):
            randomState.shuffle(Xy)
        else:
            np.random.shuffle(Xy)
    end1 = fraction * Xy.shape[0]
    Xy_1 = Xy[:end1, :]
    Xy_2 = Xy[end1:, :]
    [X_1, y_1] = separate_Xy(Xy_1)
    [X_2, y_2] = separate_Xy(Xy_2)
    return [X_1, y_1, X_2, y_2]

def shuffle_and_split_mat_rows(m, fraction, randomState = None, shuffle = True):
    if (shuffle):
        if (randomState is not None):
            randomState.shuffle(m)
        else:
            np.random.shuffle(m)
    end1 = math.ceil(fraction * m.shape[0])
    m1 = m[:end1, :]
    m2 = m[end1:, :]
    return [m1, m2]

def shuffle_and_split_lists(lists, fraction, seed = None, shuffle = True):
    orig_state = random.getstate()
    if (shuffle):
        if (seed is not None):
            random.seed(seed)
        else:
            random.shuffle(lists)
    end1 = int(math.ceil(fraction * len(lists)))
    l1 = lists[:end1]
    l2 = lists[end1:]
    random.setstate(orig_state)
    return [l1, l2]


def extract_columns_by_names(m, params, param_indices):
    cols = [False] * m.shape[1] 
    for i in range(0, len(params)):
        p = params[i]
        i = param_indices[p]
        cols[i] = True
        if (param_indices.has_key(p + "_present")):
            cols[param_indices[p + "_present"]] = True

    delete_cols = []
    for i in range(0, m.shape[1]):
        if (not cols[i]):
            delete_cols.append(i)
    
    return np.delete(m, delete_cols, axis = 1)

    """
    p = params[0]
    m_new = None
    if (param_indices.has_key(p + "_present")):
        m_new = np.matrix(m[:, param_indices[p + "_present"]]).T
        m_new = np.concatenate((m_new, np.matrix(m[:, param_indices[p]]).T), axis = 1)
    else:
        m_new = np.matrix(m[:, param_indices[p]]).T
    
    for i in range(1, len(params)):
        p = params[i]
        i = param_indices[p]
        if (param_indices.has_key(p + "_present")):
            m_new = np.concatenate((m_new, np.matrix(m[:, param_indices[p + "_present"]]).T), axis = 1)
        m_new = np.concatenate((m_new, np.matrix(m[:, i]).T), axis = 1)
    print m
    print m_new.T
    assert(m_new.shape[0] == m.shape[0])
    return m_new
    """

