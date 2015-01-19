import os
import sys
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath,'..'))
import utils
from prepare_data_set import read_data, handle_missing_data, FeatureAbsentException, prepare_data_set, normalize_data, add_offset_feature, generate_param_indices
import numpy as np

def create_tmp_folder():
    p = "/tmp/fitness"
    if (not os.path.isdir(p)):
        os.mkdir(p)

def test_read_data():
    # substitute mode
    cwd = os.path.dirname(os.path.abspath(__file__))
    infile = os.path.join(cwd, "./data","workouts_condensed_2.gz")
    x_params = ["alt(avg)","Distance"]
    y_param = "Duration"
    [X, y] = read_data(infile = infile, sport = "Running", x_params = x_params, y_param = y_param, missing_data_mode = "substitute")
    expected_y = np.matrix([[1443.0],[ 1803.0], [1672.0]])
    expected_X = np.matrix([[0,0,1,2.35], [1,98.1,1,3.35], [1,100.23,1,4.35]])
    assert(np.array_equal(y, expected_y))
    assert(np.array_equal(expected_X, X))
    
    # ignore mode
    [X, y] = read_data(infile = infile, sport = "Running", x_params = x_params, y_param = y_param, missing_data_mode = "ignore")
    expected_y = np.matrix([[ 1803.0], [1672.0]])
    expected_X = np.matrix([[98.1,3.35], [100.23,4.35]])
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
    # substitute mode
    X = np.matrix([[0,99.165,1,2.35], [1,98.1,1,3.35], [1,100.23,1,4.35]])
    mean1 = np.mean([99.165, 98.1, 100.23])
    std1 = np.std([99.165, 98.1, 100.23])
    mean2 = np.mean([2.35, 3.35, 4.35])
    std2 = np.std([2.35, 3.35, 4.35])
    expected_X = np.matrix([[0, (99.165 - mean1)/std1, 1, (2.35 - mean2) / std2], 
                    [1, (98.1 - mean1)/std1, 1, (3.35 - mean2) / std2], 
                    [1, (100.23 - mean1)/std1, 1, (4.35- mean2) / std2]])
    X = normalize_data(X, missing_data_mode = "substitute")
    assert(np.array_equal(X, expected_X))
    for i in range(1, X.shape[1],2):
        col = np.array(X[:, i])
        assert(np.allclose(np.mean(col), 0.0, rtol=1e-10, atol=1e-10))
        assert(np.allclose(np.var(col), 1.0, rtol=1e-10, atol=1e-10))

    # ignore mode
    X = np.matrix([[99.165,2.35], [98.1,3.35], [100.23,4.35]])
    mean1 = np.mean([99.165, 98.1, 100.23])
    std1 = np.std([99.165, 98.1, 100.23])
    mean2 = np.mean([2.35, 3.35, 4.35])
    std2 = np.std([2.35, 3.35, 4.35])
    expected_X = np.matrix([[(99.165 - mean1)/std1, (2.35 - mean2) / std2], 
                    [(98.1 - mean1)/std1, (3.35 - mean2) / std2], 
                    [(100.23 - mean1)/std1, (4.35- mean2) / std2]])
    X = normalize_data(X, missing_data_mode = "ignore")
    assert(np.array_equal(X, expected_X))
    for i in range(0, X.shape[1]):
        col = np.array(X[:, i])
        assert(np.allclose(np.mean(col), 0.0, rtol=1e-10, atol=1e-10))
        assert(np.allclose(np.var(col), 1.0, rtol=1e-10, atol=1e-10))


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
    outfile = "/tmp/fitness/temp.npz"
    sport = "Running"
    
    # substitute mode
    prepare_data_set(infile = infile, sport = sport, x_params = x_params, y_param = y_param, outfile = outfile, missing_data_mode = "substitute", split_fraction = 1.0, shuffle_before_split = False) # don't split
    expected_y = np.matrix([[1443.0],[ 1803.0], [1672.0]])
    mean1 = np.mean([99.165, 98.1, 100.23])
    std1 = np.std([99.165, 98.1, 100.23])
    mean2 = np.mean([2.35, 3.35, 4.35])
    std2 = np.std([2.35, 3.35, 4.35])
    expected_X = np.matrix([[1, 0, (99.165 - mean1)/std1, 1, (2.35 - mean2) / std2], 
                    [1, 1, (98.1 - mean1)/std1, 1, (3.35 - mean2) / std2], 
                    [1, 1, (100.23 - mean1)/std1, 1, (4.35- mean2) / std2]])
    expected_param_indices = {"intercept": 0, "alt(avg)_present" : 1, "alt(avg)" : 2, "Distance_present" : 3, "Distance" : 4}
    assert(os.path.isfile(outfile))
    obtained_X = np.load(outfile)["X1"]
    obtained_y = np.load(outfile)["y1"]
    assert(np.load(outfile)["X2"].shape[0] == 0)
    assert(np.load(outfile)["y2"].shape[0] == 0)
    param_indices = np.load(outfile)["param_indices"][()]
    assert(expected_param_indices == param_indices)
    assert(np.array_equal(obtained_y, expected_y))
    assert(np.allclose(expected_X, obtained_X))

    # ignore mode
    prepare_data_set(infile = infile, sport = sport, x_params = x_params, y_param = y_param, outfile = outfile, missing_data_mode = "ignore", split_fraction = 1.0, shuffle_before_split = False)
    expected_y = np.matrix([[ 1803.0], [1672.0]])
    mean1 = np.mean([98.1, 100.23])
    std1 = np.std([98.1, 100.23])
    mean2 = np.mean([3.35, 4.35])
    std2 = np.std([3.35, 4.35])
    expected_X = np.matrix([[1, (98.1 - mean1)/std1, (3.35 - mean2) / std2], 
                    [1, (100.23 - mean1)/std1, (4.35- mean2) / std2]])
    expected_param_indices = {"intercept": 0, "alt(avg)" : 1, "Distance" : 2}
    assert(os.path.isfile(outfile))
    obtained_X = np.load(outfile)["X1"]
    obtained_y = np.load(outfile)["y1"]
    assert(np.load(outfile)["X2"].shape[0] == 0)
    assert(np.load(outfile)["y2"].shape[0] == 0)
    param_indices = np.load(outfile)["param_indices"][()]
    assert(expected_param_indices == param_indices)
    print obtained_y
    print expected_y
    assert(np.array_equal(obtained_y, expected_y))
    assert(np.allclose(expected_X, obtained_X))

def test_generate_param_indices():
    # substitute mode
    x_params = ["alt(avg)","Distance"]
    assert(generate_param_indices(x_params, missing_data_mode = "substitute", with_intercept = True) 
            == {"intercept" : 0, "alt(avg)_present" : 1, "alt(avg)" : 2, "Distance_present" : 3, "Distance" : 4})
    assert(generate_param_indices(x_params, missing_data_mode = "substitute", with_intercept = False) 
            == {"alt(avg)_present" : 0, "alt(avg)" : 1, "Distance_present" : 2, "Distance" : 3})

    # ignore mode
    x_params = ["alt(avg)","Distance"]
    assert(generate_param_indices(x_params, missing_data_mode = "ignore", with_intercept = True) 
            == {"intercept" : 0, "alt(avg)" : 1, "Distance" : 2})
    assert(generate_param_indices(x_params, missing_data_mode = "ignore", with_intercept = False) 
            == {"alt(avg)" : 0, "Distance" : 1})

if __name__ == "__main__":
    test_read_data()
    test_handle_missing_data()
    test_prepare_data_set()
    test_normalize_data()
    test_add_offset_feature()
    test_generate_param_indices()
