#!/usr/bin/python

import os
import sys
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath,'..'))
from visualize import get_avg_data
import utils

def are_lists_equal(l1, l2):
    if (len(l1) != len(l2)):
        return False
    n = len(l1)
    for i in range(0, n):
        if (abs(l1[i] - l2[i]) > 1e-6):
            return False
    return True

def test_get_avg_data():
    cwd = os.path.dirname(os.path.abspath(__file__))
    infile = os.path.join(cwd, "./data","alldump1.txt")

    # positive test
    x_params = ["duration", "distance"]
    y_params = ["pace", "pace"]
    [x, y] = get_avg_data(infile, x_params, y_params)
    assert(len(x) == 2)
    assert(len(y) == 2)
    for i in range(0, len(x_params)):
        assert(len(x[i]) == 3)
        assert(len(y[i]) == 3)
    assert(y[0] == y[1])
    assert(are_lists_equal(x[0], [107041.5, 105120.25,105237.2]))
    assert(are_lists_equal(x[1],  [0.078509, 0.07995, 0.076088]))
    assert(are_lists_equal(y[0],  [7.8122315, 8.1547095, 7.8924752]))


if __name__ == "__main__":
    test_get_avg_data()
