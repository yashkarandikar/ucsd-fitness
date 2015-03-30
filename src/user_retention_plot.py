import numpy as np
import matplotlib.pyplot as plt
import utils
import predictor_duration_evolving_user as p

def get_avg_experience_per_workout_number(data, param_indices, is_active):
    exp_avgs_by_workout_number_inactive = [0.0] * 1000
    exp_counts_by_workout_number_inactive = [0.0] * 1000
    exp_avgs_by_workout_number_active = [0.0] * 1000
    exp_counts_by_workout_number_active = [0.0] * 1000
    max_workout_number_active = 0
    max_workout_number_inactive = 0
    N = data.shape[0]
    ind_u = param_indices["user_number"]
    ind_e = param_indices["experience"]
    i = 0
    while i < N:
        u = int(data[i, ind_u])
        j = 0   # workout number of this user
        while (i < N and data[i, ind_u] == u):
            e = data[i, ind_e]
            if (is_active[u]):
                exp_avgs_by_workout_number_active[j] += e
                exp_counts_by_workout_number_active[j] += 1.0
            else:
                exp_avgs_by_workout_number_inactive[j] += e
                exp_counts_by_workout_number_inactive[j] += 1.0
            j += 1
            i += 1
        if (is_active[u]):
            if (j > max_workout_number_active):
                max_workout_number_active = j
        else:
            if (j > max_workout_number_inactive):
                max_workout_number_inactive = j
    max_workout_number = min(max_workout_number_active, max_workout_number_inactive)
    max_workout_number = 50
    exp_avgs_by_workout_number_active = exp_avgs_by_workout_number_active[:max_workout_number]
    exp_counts_by_workout_number_active = exp_counts_by_workout_number_active[:max_workout_number]
    exp_avgs_by_workout_number_inactive = exp_avgs_by_workout_number_inactive[:max_workout_number]
    exp_counts_by_workout_number_inactive = exp_counts_by_workout_number_inactive[:max_workout_number]
    for i in xrange(0, max_workout_number):
        exp_avgs_by_workout_number_active[i] /= exp_counts_by_workout_number_active[i]
        exp_avgs_by_workout_number_inactive[i] /= exp_counts_by_workout_number_inactive[i]
    return exp_avgs_by_workout_number_active, exp_avgs_by_workout_number_inactive

def classify_users(data, param_indices, threshold):
    user_class_threshold = utils.parse_date_time(threshold)
    N = data.shape[0]
    ind_u = param_indices["user_number"]
    ind_t = param_indices["date-time"]
    i = 0
    is_active = []
    n_active = 0
    n_inactive = 0
    while i < N:
        u = data[i, ind_u]
        j = 0   # workout number of this user
        while (i < N and data[i, ind_u] == u):
            i += 1
        last_time = data[i - 1, ind_t]
        if (last_time > user_class_threshold):
            is_active.append(True)
            n_active += 1
        else:
            is_active.append(False)
            n_inactive += 1
    print "number of active users = ", n_active
    print "number of inactive users = ", n_inactive
    return is_active, n_active, n_inactive
        
def plot(model_file, data_file):
    model = np.load(model_file)
    sigma = model["sigma"]

    data = np.load(data_file)
    train = data["train_set"]
    param_indices = data["param_indices"][()]
    #train = train[:1000, :]
    
    print "Adding experience levels to data matrices"
    train = p.add_experience_column_to_train_set(train, sigma, param_indices)
    print param_indices

    # separate users into 2 groups
    threshold = "Apr 15, 2014 11:59 PM"
    is_active, n_active, n_inactive = classify_users(train, param_indices, threshold)

    # get avg experience after j number of workouts
    [active, inactive] = get_avg_experience_per_workout_number(train, param_indices, is_active)
    max_workout_number = len(active)

    # plot
    #print active
    #print inactive
    plt.plot(range(0, max_workout_number), active, label = "active")
    plt.plot(range(0, max_workout_number), inactive, label = "inactive")
    plt.xlabel("Workout number")
    plt.ylabel("Average experience level")
    plt.title("Threshold : %s, # Active = %d, #Inactive = %d" % (threshold, n_active, n_inactive))
    plt.legend(loc = "lower right")
    plt.show()

def stats(model_file):
    model = np.load(model_file)
    sigma = model["sigma"]
    count_by_exp = {}
    counts = []
    for s in sigma:
        for w in s:
            if (not count_by_exp.has_key(w)):
                count_by_exp[w] = 0
            count_by_exp[w] += 1
    print count_by_exp
    E = max(count_by_exp.keys())
    for i in xrange(0, E):
        counts

if __name__ == "__main__":
    #plot("model_3.npz", "../../data/all_workouts_condensed.gzfinal.npz")
    stats("model_3.npz")
