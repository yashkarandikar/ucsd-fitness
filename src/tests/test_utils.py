#!/usr/bin/python

import sys, os
import simplejson as json
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath,'..'))
import numpy as np
import utils

def test_sort_matrix_by_col():
    m = np.matrix([[3.0, 1], [1.0, 2], [2.0, 98.2]])
    col_sums = np.squeeze(np.asarray(np.sum(m, axis = 0))); row_sums = np.sort(np.squeeze(np.asarray(np.sum(m, axis = 1))))

    m = utils.sort_matrix_by_col(m, 0)
    assert(np.array_equal(m, np.matrix([[1.0, 2], [2.0, 98.2], [3.0, 1]])))
    assert(m.shape == (3, 2))
    new_col_sums = np.squeeze(np.asarray(np.sum(m, axis = 0))); new_row_sums = np.sort(np.squeeze(np.asarray(np.sum(m, axis = 1))))
    assert(np.array_equal(col_sums, new_col_sums))
    assert(np.array_equal(row_sums, new_row_sums))
    
    m = utils.sort_matrix_by_col(m, 1)
    assert(np.array_equal(m, np.matrix([[3.0, 1], [1.0, 2], [2.0, 98.2]])))
    assert(m.shape == (3, 2))
    new_col_sums = np.squeeze(np.asarray(np.sum(m, axis = 0))); new_row_sums = np.sort(np.squeeze(np.asarray(np.sum(m, axis = 1))))
    assert(np.array_equal(col_sums, new_col_sums))
    assert(np.array_equal(row_sums, new_row_sums))

    x = np.random.randint(2, 100)
    y = np.random.randint(2, 100)
    m = np.random.rand(x, y)
    col_sums = np.squeeze(np.asarray(np.sum(m, axis = 0))); row_sums = np.sort(np.squeeze(np.asarray(np.sum(m, axis = 1))))
    for i in range(0, y):
        m = utils.sort_matrix_by_col(m, i)
        assert(m.shape == (x, y))
        new_col_sums = np.squeeze(np.asarray(np.sum(m, axis = 0))); new_row_sums = np.sort(np.squeeze(np.asarray(np.sum(m, axis = 1))))
        assert(np.allclose(col_sums, new_col_sums, 1e-12, 1e-12))
        assert(np.allclose(row_sums, new_row_sums, 1e-12, 1e-12))
        for j in range(0, x - 1):
            assert(m[j, i] <= m[j+1, i])

def test_combine_Xy():
    x = np.random.randint(2, 100)
    y = np.random.randint(2, 100)
    X = np.random.rand(x, y)
    Y = np.random.rand(x, 1)
    sumsX = np.squeeze(np.asarray(np.sum(X, axis = 1)))
    sumsY = np.squeeze(np.asarray(np.sum(Y, axis = 1)))
    sums = np.sort(np.add(sumsX, sumsY))
    XY = utils.combine_Xy(X, Y)
    sumsXY = np.sort(np.squeeze(np.asarray(np.sum(XY, axis = 1))))
    if (not np.array_equal(sumsXY, sums)):
        print "X = ", X
        print "Y = ", Y
        print "sumsX = ", sumsX
        print "sumsY = ", sumsY
        print "sums = sumsX + sumsY = ", sums
        print "XY = ", XY
        print "sumsXY  = ", sumsXY
    assert(np.array_equal(sumsXY, sums))

def test_separate_Xy():
    x = np.random.randint(2, 100)
    y = np.random.randint(2, 100)
    X = np.random.rand(x, y)
    Y = np.random.rand(x, 1)
    XY = utils.combine_Xy(X, Y)
    [X_sep, Y_sep] = utils.separate_Xy(XY)
    assert(np.array_equal(X, X_sep))
    assert(np.array_equal(Y, Y_sep))

def test_sort_data_by_col():
    x = np.random.randint(2, 100)
    y = np.random.randint(2, 100)
    m = np.random.rand(x, y)
    col_sums = np.squeeze(np.asarray(np.sum(m, axis = 0))); row_sums = np.sort(np.squeeze(np.asarray(np.sum(m, axis = 1))))
    [X, Y] = utils.separate_Xy(m)
    for i in range(0, y):
        [X_sort, Y_sort] = utils.sort_data_by_col(X, Y, i)
        XY_sort = utils.combine_Xy(X_sort, Y_sort)
        assert(XY_sort.shape == (x, y))
        new_col_sums = np.squeeze(np.asarray(np.sum(XY_sort, axis = 0))); new_row_sums = np.sort(np.squeeze(np.asarray(np.sum(XY_sort, axis = 1))))
        assert(np.allclose(col_sums, new_col_sums, 1e-12, 1e-12))
        assert(np.allclose(row_sums, new_row_sums, 1e-12, 1e-12))
        for j in range(0, x - 1):
            assert(XY_sort[j, i] <= XY_sort[j+1, i])

def test_remove_rows_by_condition():
    n = 10
    m = np.matrix(np.random.randint(low = 0, high = 100000, size=(n, n)))
    cols = range(0, n)
    lower_bounds = [0] * n
    upper_bounds = [100000] * n
    m2 = utils.remove_rows_by_condition(m, cols, lower_bounds, upper_bounds)
    m_sorted = utils.sort_matrix_by_col(m, 0) 
    m2_sorted = utils.sort_matrix_by_col(m2, 0)
    print m_sorted
    print m2_sorted
    print m_sorted - m2_sorted
    assert(np.array_equal(m_sorted, m2_sorted))

    m = np.matrix([[1.0, 2.0, 300.0], [5.0, -2.32, 9.99], [200.0, 32.32, 1.2]])
    cols = [0, 2]
    lower_bounds = [float("-inf"), 0.0]
    upper_bounds = [10.0, 10.0]
    m2 = utils.remove_rows_by_condition(m, cols, lower_bounds, upper_bounds)
    m_expected = np.matrix([[5.0, -2.32, 9.99]])
    assert(np.array_equal(m2, m_expected))

def test_shuffle_and_split_Xy():
    X = np.matrix(np.random.rand(100, 100))
    y = np.matrix(np.random.rand(100, 1))
    [X1, y1, X2, y2] = utils.shuffle_and_split_Xy(X, y, fraction = 0.77)
    assert(X1.shape[0] == 77)
    assert(y1.shape[0] == 77)
    assert(X2.shape[0] == 23)
    assert(y2.shape[0] == 23)

    nrows = np.random.randint(100)
    ncols = np.random.randint(100)
    X = np.matrix(np.random.rand(nrows, ncols))
    y = np.matrix(np.random.rand(nrows, 1))
    [X1, y1, X2, y2] = utils.shuffle_and_split_Xy(X, y, fraction = 0.51)
    assert(X1.shape[0] + X2.shape[0] == nrows)
    assert(X1.shape[0] == y1.shape[0])
    assert(X2.shape[0] == y2.shape[0])
    assert(X1.shape[1] == ncols)
    assert(X2.shape[1] == ncols)

def test_extract_columns_by_names():
    # ignore mode
    m = np.matrix([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    param_indices = {"intercept": 0, "Distance" : 1, "hr(avg)" : 2, "Duration" : 3, "Calories" : 4}
    m2 = utils.extract_columns_by_names(m, ["Distance", "Calories"], param_indices)
    assert(np.array_equal(m2, np.matrix([[2, 5], [2, 5], [2, 5]])))

    # substitute mode
    m = np.matrix([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    param_indices = {"intercept": 0, "Distance_present" : 1, "Distance" : 2, "Duration_present" : 3, "Duration" : 4}
    m2 = utils.extract_columns_by_names(m, ["Distance", "Duration"], param_indices)
    assert(np.array_equal(m2, np.matrix([[2, 3, 4, 5], [2, 3, 4, 5], [2, 3, 4, 5]])))

if __name__ == "__main__":
    test_sort_matrix_by_col()
    test_combine_Xy()
    test_separate_Xy()
    test_sort_data_by_col()
    test_remove_rows_by_condition()
    test_shuffle_and_split_Xy()
    test_extract_columns_by_names()
