from utils import json_to_dicts, get_user_id_from_filename
import os
import time

class User(object):
    def __init__(self, uid, n_workouts):
        self.uid = uid
        self.n_workouts = n_workouts

class Workout(object):
    attr = {'UID' : 8, 'Distance' : 9, 'Max. Speed : ' : 11, 'Avg. Speed' : 11, 'Calories' : 9, 'Max. Heart Rate' : 16, 
            'Weather' : 15, 'Total Ascent' : 14, 'Total Descent' : 14, 'Duration' : 9, 'Avg. Heart Rate' : 16, 
            'Cadence' : 8, 'date-time' : 15}

    def __init__(self, uid, workout_dict):
        self.uid = uid
        self.workout_dict = workout_dict
        self.workout_dict['UID'] = str(uid)
    
    @staticmethod
    def attribute_str():
        s = ""
        for a in Workout.attr.keys():
            tmp = "{s:<" + str(Workout.attr[a]) + "s}"
            s = s + tmp.format(s=a)
        return s

    def __str__(self):
        values = ""
        for a in Workout.attr:
            curr = "{v:<" + str(Workout.attr[a]) + "s}"
            if (self.workout_dict.has_key(a)):
                curr = curr.format(v=self.workout_dict[a])
            else:
                curr = curr.format(v="none")
            values = values + curr
        return values

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
    workout_dicts = get_data_for_user(uid)
    workouts = []
    for w in workout_dicts:
        workouts.append(Workout(uid, w))
        print w
    workouts.sort(key=lambda x: x.workout_dict['date-time'])
    print Workout.attribute_str()
    for w in workouts:
        print w 


if __name__ == "__main__":
    t1 = time.time()
    get_user_stats()
    t2 = time.time()
    #print Workout.attribute_str()
    print "Time taken = " + str(t2 - t1)
