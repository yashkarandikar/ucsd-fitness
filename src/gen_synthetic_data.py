import numpy as np
import gzip
from prepare_data_set import add_offset_feature
import utils
from datetime import datetime

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

def for_baseline_model():
    U = 4
    Nu = [10, 21, 4, 32]
    uids = [20000, 40000, 10000, 30000]
    sport = "Running"
    
    randomState = np.random.RandomState(12345)
    f = gzip.open("synth_baseline_model.gz", "w")

    theta = randomState.randint(low = 100, high = 10000, size = U)
    print theta
    data = []
    for u in xrange(0, U):
        nu = Nu[u]
        v = theta[u]
        uid = uids[u]
        for i in xrange(0, nu):
            d = randomState.randint(low = 1, high = 10)
            t = v * d
            w = {"Distance" : d, "Duration" : t, "user_id" : str(uid), "sport" : sport}
            w_json = utils.dict_to_json(w)
            f.write(w_json + "\n")
    
    f.close()

def for_user_model():
    U = 4
    Nu = [10, 21, 14, 32]
    uids = [20000, 40000, 10000, 30000]
    sport = "Running"
    
    randomState = np.random.RandomState(12345)
    f = gzip.open("synth_user_model.gz", "w")

    theta = randomState.rand(U + 3)    
    alpha_all = theta[-3]
    theta_0 = theta[-2]
    theta_1 = theta[-1]
    data = []
    for u in xrange(0, U):
        nu = Nu[u]
        alpha = theta[u]
        uid = uids[u]
        for i in xrange(0, nu):
            d = randomState.randint(low = 1, high = 10)
            t = (alpha + alpha_all) * (theta_0 + theta_1 * d) * 3600.0
            w = {"Distance" : d, "Duration" : t, "user_id" : str(uid), "sport" : sport}
            w_json = utils.dict_to_json(w)
            f.write(w_json + "\n")
    print theta
    
    f.close()

def for_evolving_user_model():
    U = 4
    E = 3
    Nu = [10, 21, 14, 32]
    uids = sorted([20000, 40000, 10000, 30000])
    print uids
    sport = "Running"

    sigma = np.zeros((U, max(Nu)))
    
    randomState = np.random.RandomState(12345)
    f = gzip.open("synth_evolving_user_model.gz", "w")

    #theta = randomState.rand(U * E + E + 2)
    theta_0 = randomState.rand()
    theta_1 = randomState.rand()
    alpha = np.array(sorted(list(randomState.rand(E)), reverse = True))   # per user alpha for each experience
    data = []
    theta = []
    for u in xrange(0, U):
        nu = Nu[u]
        #alpha_u = np.sort(randomState.rand(E))   # per user alpha for each experience
        alpha_u = np.array(sorted(list(randomState.rand(E)), reverse = True))   # per user alpha for each experience
        #print alpha_u
        theta = theta + list(alpha_u)
        uid = uids[u]
        dts = np.sort(randomState.randint(low = 1000000000, high = 1100000000, size = (nu)))
        exp_levels = np.sort(randomState.randint(low = 0, high = E, size = (nu)))
        for i in xrange(0, nu):
            d = randomState.randint(low = 1, high = 10)
            e = exp_levels[i]
            sigma[u, i] = e
            a_e = alpha[e]
            a_ue = alpha_u[e]
            print "u = %d, e = %d, alpha_ue = %f" % (u, e, a_ue)
            t = (a_e + a_ue) * (theta_0 + theta_1 * d) * 3600.0
            #print "a term = " + str((a_e + a_ue)) + " theta term = " + str(theta_0 + theta_1 * d)
            dt = datetime.fromtimestamp(dts[i]).strftime('%b %d, %Y %I:%M %p')
            w = {"Distance" : d, "Duration" : t, "user_id" : str(uid), "sport" : sport, "date-time" : dt, "experience" : e}
            w_json = utils.dict_to_json(w)
            f.write(w_json + "\n")
            data.append([u, uid, d, t, dts[i]])
    
    f.close()
    data = np.matrix(data)

    sigma_list = [list(sigma[i, :]) for i in range(0, U)]

    theta += list(alpha)
    theta += [theta_0]
    theta += [theta_1]
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    print "Actual theta : ", theta
    #print "Actual sigma : ", sigma_list
    #print "Data : ", data


if __name__ == "__main__":
    for_evolving_user_model()
    #for_user_model()
