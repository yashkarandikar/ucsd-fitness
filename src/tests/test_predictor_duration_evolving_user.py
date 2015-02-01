import os
import sys
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath,'..'))
import predictor_duration_evolving_user as p
import numpy as np

def test_find_best_path_DP():
    # take a small matrix, do a brute force on it, and see if DP returns the same solution
    M = np.matrix([[2.0, 5.0, 4.0, 5], [1.0, 3.0, 1.0, 2.0], [3.0, 2.0, 4.0, 1.0], [4.0, 3.0, 4.0, 5.0]])
    leastError, path = p.find_best_path_DP(M)
    print leastError
    print path
    assert(path == [1, 1, 1, 2])    # note that this is the sequence of experience levels, not the errors
    assert(leastError == 6)

if __name__ == "__main__":
    test_find_best_path_DP()
