import os
import sys
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath,'..'))
import predictor_duration_user as p
import numpy as np

def test_E():
    data = np.matrix([[0, 1000021, 20.0, 1000.0], [0, 1000021, 15.0, 1000.0], [1, 3213223, 10.0, 800.0], [1, 3213223, 12.0, 900.0]])
    theta = np.array([0.10, 0.20, 100, 2000])
    assert(p.E(theta, data) == 38835000)

def test_Eprime():
    data = np.matrix([[0, 1000021, 20.0, 1000.0], [0, 1000021, 15.0, 1000.0], [1, 3213223, 10.0, 800.0], [1, 3213223, 12.0, 900.0]])
    data = np.asarray(data)
    theta = np.array([0.10, 0.20, 100, 2000])
    theta_new = p.Eprime(theta, data)
    expected_theta_new = np.asarray([362404000, 318388000, 3860, 49766])
    assert(np.array_equal(expected_theta_new, theta_new))

def test_Eprime_slow():
    data = np.matrix([[0, 1000021, 20.0, 1000.0], [0, 1000021, 15.0, 1000.0], [1, 3213223, 10.0, 800.0], [1, 3213223, 12.0, 900.0]])
    data = np.asarray(data)
    theta = np.array([0.10, 0.20, 100, 2000])
    theta_new = p.Eprime_slow(theta, data)
    expected_theta_new = np.asarray([362404000, 318388000, 3860, 49766])
    assert(np.array_equal(expected_theta_new, theta_new))


def test_shuffle_and_split_data_by_user():
    pass

def test_add_user_number_column():
    pass

def test_compute_stats():
    pass

if __name__ == "__main__":
    test_E()    
    test_Eprime()
    test_Eprime_slow()
