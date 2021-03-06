import os
import sys
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath,'..'))
import predictor_duration_evolving_user as p
import numpy as np
import scipy
import pyximport; pyximport.install()
from predictor_duration_evolving_user_pyx import F_pyx, Fprime_pyx

def test_find_best_path_DP():
    # take a small matrix, do a brute force on it, and see if DP returns the same solution
    M = np.matrix([[2.0, 5.0, 4.0, 5], [1.0, 3.0, 1.0, 2.0], [3.0, 2.0, 4.0, 1.0], [4.0, 3.0, 4.0, 5.0]])
    leastError, path = p.find_best_path_DP(M)
    print leastError
    print path
    assert(path == [1, 1, 1, 2])    # note that this is the sequence of experience levels, not the errors
    assert(leastError == 6)

"""
def test_Fprime():
    E = 3
    U = 2
    data = np.matrix([[0, 1000021, 20.0, 1000.0, 1000000], [0, 1000021, 15.0, 1000.0, 1000001], [1, 3213223, 10.0, 800.0, 2000000], [1, 3213223, 12.0, 900.0, 2000001]])
    randomState = np.random.RandomState(12345)
    Nu = 2
    sigma = [[0, 1], [1, 2]]
    lam = 0.221
    #theta = np.array(randomState.rand(U * E + E + 2))
    theta = np.array(np.ones(U * E + E + 2) * 100)
    print theta
    our_grad = np.linalg.norm(p.Fprime(theta, data, lam, E, sigma), ord = 2)
    numerical = np.linalg.norm(scipy.optimize.approx_fprime(theta, p.F, np.sqrt(np.finfo(np.float).eps), data, lam, E, sigma), ord = 2)
    ratio = our_grad / numerical
    print "our = ", our_grad
    print "numerical = ", numerical
    print "Ratio = ", ratio
    assert(abs(1.0 - ratio) < 1e-5)
"""

def test_Fprime_pyx():
    E = 3
    U = 2
    data = np.matrix([[0, 1000021, 20.0, 1000.0, 1000000], [0, 1000021, 15.0, 1000.0, 1000001], [1, 3213223, 10.0, 800.0, 2000000], [1, 3213223, 12.0, 900.0, 2000001]])
    randomState = np.random.RandomState(12345)
    Nu = 2
    sigma = [[0, 1], [1, 2]]
    print sigma
    lam = 2232.221
    #theta = np.array(randomState.rand(U * E + E + 2))
    theta = np.array(np.ones(U * E + E + 2) * 100)
    print theta
    our_grad = np.linalg.norm(Fprime_pyx(theta, data, lam, E, sigma), ord = 2)
    numerical = np.linalg.norm(scipy.optimize.approx_fprime(theta, F_pyx, np.sqrt(np.finfo(np.float).eps), data, lam, E, sigma), ord = 2)
    ratio = our_grad / numerical
    print "our = ", our_grad
    print "numerical = ", numerical
    print "Ratio = ", ratio
    assert(abs(1.0 - ratio) < 1e-5)

def test_add_experience_column_to_train_set():
    U = 4
    E = 3
    Nu = [2, 4, 2, 6]
    uids = sorted([20000, 40000, 10000, 30000])
    sport = "Running"
    sigma = []
    randomState = np.random.RandomState(12345)
    data = []
    param_indices = {"user_number":0, "user_id":1, "Distance":2, "Duration":3, "date-time" : 4}
    all_exp = []
    for u in range(0, U):
        nu = Nu[u]
        dts = np.sort(randomState.randint(low = 1000000000, high = 1100000000, size = (nu)))
        exp_levels = np.sort(randomState.randint(low = 0, high = E, size = (nu)))
        sigma.append([0] * nu)
        for i in range(0, nu):
            data.append([u, uids[u], 0.0, 0.0, dts[i]])
            sigma[u][i] = exp_levels[i]
            all_exp.append(exp_levels[i])
    data = np.matrix(data)
    obtained_data = p.add_experience_column_to_train_set(data, sigma, param_indices)
    assert(param_indices == {"user_number":0, "user_id":1, "Distance" : 2, "Duration" : 3, "date-time" : 4, "experience" : 5})
    expected_data = np.concatenate((data, np.matrix(all_exp).T), axis = 1)
    assert(np.array_equal(obtained_data, expected_data))

def test_add_experience_column_to_test_set():
    train = np.matrix([[0, 10000, 0.0, 0.0, 10000, 0], [0, 10000, 0.0, 0.0, 10050, 1],
            [1, 20000, 0.0, 0.0, 10000, 1], [1, 20000, 0.0, 0.0, 10050, 2]])
    val = np.matrix([[0, 10000, 1.0, 2.0, 10020], [0, 10000, 1.0, 2.0, 10040],
            [1, 20000, 3.0, 4.0, 9999], [1, 20000, 321.32, 321.321, 10100]])
    param_indices = {"user_number":0, "user_id":1, "Distance":2, "Duration":3, "date-time":4, "experience" : 5}
    val_obtained = p.add_experience_column_to_test_set(val, train, param_indices, mode = "random")
    val_expected = np.matrix([[0, 10000, 1.0, 2.0, 10020, 0], [0, 10000, 1.0, 2.0, 10040, 1],
            [1, 20000, 3.0, 4.0, 9999, 1], [1, 20000, 321.32, 321.321, 10100, 2]])
    assert(np.array_equal(val_obtained, val_expected))

if __name__ == "__main__":
    test_find_best_path_DP()
    #test_Fprime()
    test_Fprime_pyx()
    test_add_experience_column_to_train_set()
    test_add_experience_column_to_test_set()
