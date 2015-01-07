#!/usr/bin/python

import os
import sys
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath,'..'))
from plot_data import get_data
import utils

def are_lists_equal(l1, l2):
    if (len(l1) != len(l2)):
        return False
    n = len(l1)
    for i in range(0, n):
        if (abs(l1[i] - l2[i]) > 1e-6):
            return False
    return True

def test_get_data():
    cwd = os.path.dirname(os.path.abspath(__file__))
    infile = os.path.join(cwd, "./data","workouts_condensed.gz")

    # positive test
    x_params = ["duration(avg)", "Distance"]
    y_params = ["pace(avg)", "pace(avg)"]
    sports = ["Running", "Running"]
    objs = get_data(infile, x_params, y_params, sports)
    assert(len(objs) == 2)
    for i in range(0, len(x_params)):
        assert(len(objs[i].xvals) == 3)
        assert(len(objs[i].yvals) == 3)
    assert(objs[0].yvals == objs[1].yvals)
    assert(are_lists_equal(objs[0].xvals, [103077.0, 103127.8, 105237.2]))
    assert(are_lists_equal(objs[1].xvals,  [2.35, 3.35, 4.35]))
    assert(are_lists_equal(objs[0].yvals,  [7.812231, 8.15471, 7.892475]))
    assert(not objs[0].empty())
    assert(not objs[1].empty())

    # negative test
    x_params = ["Distance"]
    y_params = ["No such param"]
    sports = ["Running"]
    objs = get_data(infile, x_params, y_params, sports)
    assert(len(objs) == 1)
    assert(objs[0].empty())

'''
def test_sort_avg_data():
    x = [[], []]; y = [[], []]
    x[0] = [1.1, 2.2, 3.3]
    y[0] = [5, 6, 7]
    x[1] = [2.11, 3.12, 1.22]
    y[1] = [100, 200, 300]
    [x, y] = sort_avg_data(x, y)
    assert(len(x) == 2)
    assert(len(y) == 2)
    assert(are_lists_equal(x[0], [1.1, 2.2, 3.3]))
    assert(are_lists_equal(y[0], [5, 6, 7]))
    assert(are_lists_equal(x[1], [1.22, 2.11, 3.12]))
    assert(are_lists_equal(y[1], [300, 100, 200]))
'''

if __name__ == "__main__":
    test_get_data()
