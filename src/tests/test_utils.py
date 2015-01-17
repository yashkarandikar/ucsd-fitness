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


if __name__ == "__main__":
    test_sort_matrix_by_col()
    test_combine_Xy()
    test_separate_Xy()
    test_sort_data_by_col()
