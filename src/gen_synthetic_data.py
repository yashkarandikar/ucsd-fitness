import numpy as np
import gzip
from prepare_data_set import add_offset_feature
import utils

def for_linear_model():
    theta = np.matrix([[0.1, 0.2]]).T
    X = np.matrix(np.random.rand(1000, 1))
    X = add_offset_feature(X)
    print X.shape
    noise =  np.matrix(np.random.normal(0, 0.1, 1000)).T
    Y = X.dot(theta) + noise
    print Y.shape
    np.save("X_synthetic_direct.npy", X)
    np.save("Y_synthetic_direct.npy", Y)

    with gzip.open("synthetic_workouts.gz", "w") as f:
        for i in range(0, 1000):
            d = {}
            d["Distance"] = 100 * X[i, 1]
            d["Duration"] = 100 * Y[i, 0]
            s = utils.dict_to_json(d)
            f.write(s + "\n")

def for_user_model():
    U = 4
    Nu = [10, 21, 4, 32]
    uids = [20000, 40000, 10000, 30000]
    sport = "Running"
    
    randomState = np.random.RandomState(12345)
    f = gzip.open("synth1.gz", "w")

    theta = randomState.rand(U + 2)    
    theta_0 = theta[-2]
    theta_1 = theta[-1]
    data = []
    for u in xrange(0, U):
        nu = Nu[u]
        alpha = theta[u]
        uid = uids[u]
        for i in xrange(0, nu):
            d = randomState.randint(low = 1, high = 10)
            t = alpha * (theta_0 + theta_1 * d) * 3600.0
            w = {"Distance" : d, "Duration" : t, "user_id" : str(uid), "sport" : sport}
            w_json = utils.dict_to_json(w)
            f.write(w_json + "\n")
    
    f.close()

if __name__ == "__main__":
    for_user_model()
