from utils import json_to_dicts, get_user_id_from_filename
import os
import time

class User(object):
    def __init__(self, uid, n_workouts):
        self.uid = uid
        self.n_workouts = n_workouts

def get_data():
    cwd = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(cwd, "..","data","endoMondo5000")
    files = os.listdir(folder)
    #full_files = [os.path.join(folder, f) for f in files]
    users = []
    for f in files:
        uid = get_user_id_from_filename(f)
        f_path = os.path.join(folder, f)
        #data[uid] = json_to_dicts(f_path)
        users.append(User(uid, len(json_to_dicts(f_path))))
    return users

def get_data_for_user(user_id):
    cwd = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(cwd, "..","data","endoMondo5000")
    f_path = os.path.join(folder, str(user_id) + ".txt")
    return json_to_dicts(f_path)

def get_user_stats():
    # get users
    users = get_data()
    users.sort(key=lambda x: x.n_workouts, reverse=True)
    print "Users with highest number of workouts:\n"
    print "{uid:>12s}{n_workouts:>12s}".format(uid="User ID", n_workouts = "# workouts")
    for u in users[:10]:
        print "{uid:12d}{n_workouts:12d}".format(uid=u.uid, n_workouts=u.n_workouts)

    # get user with max workouts
    uid = users[0].uid
    print "\nUser with highest number of workouts = " + str(uid)
    workouts = get_data_for_user(uid)


if __name__ == "__main__":
    t1 = time.time()
    get_user_stats()
    t2 = time.time()
    print "Time taken = " + str(t2 - t1)
