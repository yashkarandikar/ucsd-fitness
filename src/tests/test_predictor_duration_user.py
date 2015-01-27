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

    uids = np.matrix([100000] * 25 + [1000001] * 25 + [23123321] * 25 + [542132131] * 25).T
    uins = np.matrix([0] * 25 + [1] * 25 + [2] * 25 + [3] * 25).T
    theta = np.array([0.1, 0.2, 1.12, 2.3, 2.1, 32.76])
    data = np.random.rand(100, 2)
    data = np.concatenate((uins, uids, data), axis = 1)
    slow = p.Eprime_slow(theta, data)
    fast = p.Eprime(theta, data)
    assert(np.allclose(slow, fast, rtol=1e-08, atol=1e-08))
    
    data = np.matrix([[0, 1000021, 20.0, 1000.0], [0, 1000021, 15.0, 1000.0], [1, 3213223, 10.0, 800.0], [1, 3213223, 12.0, 900.0]])
    data = np.asarray(data)
    theta = np.array([0.10, 0.20, 100, 2000])
    theta_new = p.Eprime_slow(theta, data)
    expected_theta_new = np.asarray([362404000, 318388000, 3860, 49766])
    assert(np.array_equal(expected_theta_new, theta_new))

def test_shuffle_and_split_data_by_user():
    data = np.matrix([[0, 1000021, 20.0, 1000.0], [1, 3213223, 15.0, 1000.0], [1, 3213223, 10.0, 800.0], [1, 3213223, 12.0, 900.0]])
    expected_d1 = data[1:3, :]
    expected_d2 = data[3, :]
    [d1, d2] = p.shuffle_and_split_data_by_user(data)
    assert(np.array_equal(d1, expected_d1))
    assert(np.array_equal(d2, expected_d2))
    [d1, d2] = p.shuffle_and_split_data_by_user_slow(data)
    assert(np.array_equal(d1, expected_d1))
    assert(np.array_equal(d2, expected_d2))


def test_get_user_count():
    data = np.matrix([[0, 1000021, 20.0, 1000.0], [1, 3213223, 15.0, 1000.0], [1, 3213223, 10.0, 800.0], [1, 3213223, 12.0, 900.0]])
    assert(p.get_user_count(data) == 2)
    data = [[0, 1000021, 20.0, 1000.0], [1, 3213223, 15.0, 1000.0], [1, 3213223, 10.0, 800.0], [1, 3213223, 12.0, 900.0]]
    assert(p.get_user_count(data) == 2)
    data = np.asarray(np.matrix([[0, 1000021, 20.0, 1000.0], [1, 3213223, 15.0, 1000.0], [1, 3213223, 10.0, 800.0], [1, 3213223, 12.0, 900.0]]))
    assert(p.get_user_count(data) == 2)

def test_add_user_number_column():
    data = [[3213223, 15.0, 1000.0], [9999993, 10.0, 800.0], [3213223, 12.0, 900.0], [1000021, 20.0, 1000.0]]
    p.add_user_number_column(data)
    expected_data = [[0, 1000021, 20.0, 1000.0], [1, 3213223, 15.0, 1000.0], [1, 3213223, 12.0, 900.0],  [2, 9999993, 10.0, 800.0]]
    assert(data == expected_data)

def test_compute_stats():
    data = np.matrix([[0, 1000000, 20.0, 2010.0], [0, 1000000, 15, 1510], [1, 7000000, 10, 2020.0], [1, 7000000, 12.0, 2420.0]])
    theta = np.array([0.1, 0.2, 100.0, 1000.0])
    [mse, var, fvu, r2] = p.compute_stats(data, theta)
    assert(mse == 0.0)
    assert(var == np.var([2010.0, 1510.0, 2020.0, 2420.0]))
    assert(fvu == 0.0)
    assert(r2 == 1.0)

if __name__ == "__main__":
    test_get_user_count()
    test_E()    
    test_Eprime()
    test_shuffle_and_split_data_by_user()
    test_add_user_number_column()
    test_compute_stats()
