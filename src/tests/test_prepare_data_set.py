import os
import sys
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath,'..'))
import utils
from prepare_data_set import read_data, handle_missing_data, FeatureAbsentException, prepare_data_set, normalize_data, add_offset_feature
import numpy as np

def create_tmp_folder():
    p = "/tmp/fitness"
    if (not os.path.isdir(p)):
        os.mkdir(p)

def test_read_data():
    cwd = os.path.dirname(os.path.abspath(__file__))
    infile = os.path.join(cwd, "./data","workouts_condensed_2.gz")
    x_params = ["alt(avg)","Distance"]
    y_param = "Duration"
    [X, y] = read_data(infile = infile, x_params = x_params, y_param = y_param)
    expected_y = np.matrix([[1443.0],[ 1803.0], [1672.0]])
    expected_X = np.matrix([[0,0,1,2.35], [1,98.1,1,3.35], [1,100.23,1,4.35]])
    assert(np.array_equal(y, expected_y))
    assert(np.array_equal(expected_X, X))

def test_handle_missing_data():
    # test case where missing data is subsituted correctly
    X = np.matrix([[0,0,1,2.35], [1,98.1,1,3.35], [1,100.23,1,4.35]])
    expected_X = np.matrix([[0,(98.1+100.23)/2.0,1,2.35], [1,98.1,1,3.35], [1,100.23,1,4.35]])
    handle_missing_data(X, ["p1","p2"])
    assert(np.array_equal(X, expected_X))
    
    # case where a feature is entirely missing
    X = np.matrix([[0,0,1,2.35], [0,0,1,3.35], [0,0,1,4.35]])
    try:
        handle_missing_data(X, ["p1","p2"])
        assert False
    except FeatureAbsentException as e:
        assert(e.feature == "p1")

    # case where no data is missing
    X = np.matrix([[1,110.2,1,2.35], [1,98.1,1,3.35], [1,100.23,1,4.35]])
    expected_X = np.matrix([[1,110.2,1,2.35], [1,98.1,1,3.35], [1,100.23,1,4.35]])
    handle_missing_data(X, ["p1","p2"])
    assert(np.array_equal(X, expected_X))

def test_normalize_data():
    X = np.matrix([[0,99.165,1,2.35], [1,98.1,1,3.35], [1,100.23,1,4.35]])
    mean1 = np.mean([99.165, 98.1, 100.23])
    std1 = np.std([99.165, 98.1, 100.23])
    mean2 = np.mean([2.35, 3.35, 4.35])
    std2 = np.std([2.35, 3.35, 4.35])
    expected_X = np.matrix([[0, (99.165 - mean1)/std1, 1, (2.35 - mean2) / std2], 
                    [1, (98.1 - mean1)/std1, 1, (3.35 - mean2) / std2], 
                    [1, (100.23 - mean1)/std1, 1, (4.35- mean2) / std2]])
    X = normalize_data(X)
    assert(np.array_equal(X, expected_X))

def test_add_offset_feature():
    X = np.matrix([[5.63, 43.21, 431.2], [321.3, 32.4, 9.1], [3.3, 114.3, 98.98]])
    expected_X = np.matrix([[1.0, 5.63, 43.21, 431.2], [1.0, 321.3, 32.4, 9.1], [1.0, 3.3, 114.3, 98.98]])
    X = add_offset_feature(X)
    assert(np.array_equal(X, expected_X))

def test_prepare_data_set():
    cwd = os.path.dirname(os.path.abspath(__file__))
    infile = os.path.join(cwd, "./data","workouts_condensed_2.gz")
    x_params = ["alt(avg)","Distance"]
    y_param = "Duration"
    prepare_data_set(infile = infile, x_params = x_params, y_param = y_param, outfile_base = "/tmp/fitness/")
    expected_y = np.matrix([[1443.0],[ 1803.0], [1672.0]])
    #expected_X = np.matrix([[1,0,(98.1 + 100.23)/2.0,1,2.35], [1,1,98.1,1,3.35], [1,1,100.23,1,4.35]])
    mean1 = np.mean([99.165, 98.1, 100.23])
    std1 = np.std([99.165, 98.1, 100.23])
    mean2 = np.mean([2.35, 3.35, 4.35])
    std2 = np.std([2.35, 3.35, 4.35])
    expected_X = np.matrix([[1, 0, (99.165 - mean1)/std1, 1, (2.35 - mean2) / std2], 
                    [1, 1, (98.1 - mean1)/std1, 1, (3.35 - mean2) / std2], 
                    [1, 1, (100.23 - mean1)/std1, 1, (4.35- mean2) / std2]])
    assert(os.path.isfile("/tmp/fitness/X.npy"))
    assert(os.path.isfile("/tmp/fitness/y.npy"))
    obtained_X = np.load("/tmp/fitness/X.npy")
    obtained_y = np.load("/tmp/fitness/y.npy")
    assert(np.array_equal(obtained_y, expected_y))
    assert(np.allclose(expected_X, obtained_X))


if __name__ == "__main__":
    test_read_data()
    test_handle_missing_data()
    test_prepare_data_set()
    test_normalize_data()
    test_add_offset_feature()
