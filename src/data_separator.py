from utils import json_to_dicts, get_user_id_from_filename
import os
import time
import argparse
import simplejson as json
import gzip
import shutil

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

def get_data(data_folder):
    cwd = os.path.dirname(os.path.abspath(__file__))
    #folder = os.path.join(cwd, "..","data","endoMondo5000")
    subfolders = os.listdir(data_folder)
    #full_files = [os.path.join(folder, f) for f in files]
    #print files
    users = []
    full_files = []
    for sf in subfolders:
        sf_path = os.path.join(data_folder, sf)
        files = os.listdir(sf_path)
        for f in files:
            f_path = os.path.join(data_folder, sf, f)
            #print f_path
            uid = get_user_id_from_filename(f_path)
            users.append(User(uid, len(json_to_dicts(f_path))))
    return users

def get_data_for_user(data_folder, user_id):
    cwd = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(cwd, "..","data","endoMondo5000")
    f_path = os.path.join(data_folder, str(user_id) + ".txt")
    return json_to_dicts(f_path)



def get_user_stats(data_folder):
    # get users
    users = get_data(data_folder)
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

def get_user_outfolder(outfolder, user_id):
    # returns path to a folder under outfolder, named with the first 3 digits of user id
    return os.path.join(outfolder, str(user_id)[:3])

def add_workout_to_user_file(workout_dict, outfolder):
    user_id = workout_dict["user_id"]

    # creates a file for the user (if does not already exist) and adds a workout in JSON format
    folder = get_user_outfolder(outfolder, user_id)
    if (not os.path.isdir(folder)):
        os.mkdir(folder)
    filepath = os.path.join(folder, str(user_id) + ".txt")

    # add to user's file
    workout_str = json.dumps(workout_dict)
    with open(filepath, 'a') as f:
        f.write(workout_str + "\n")
    return True

def separate_users(infile, outfolder):
    # reads data from ONE big gzipped file and writes data of each user to separate file
    if (not os.path.isfile(infile)):
        raise Exception("Gzipped file not found")

    # delete old output folder, if any and create new one
    if (os.path.isdir(outfolder)):
        print "removing folder " + outfolder
        shutil.rmtree(outfolder)
    print "creating folder " + outfolder
    os.mkdir(outfolder)

    # do not use json_to_dicts() here since it will attempt to load all dicts into memory
    i = 0
    users = {}
    with gzip.open(infile) as f:
        for line in f:
            # each line is in json format, check user and add to appropriate file
            d = json.loads(line)        # convert json line to dictionary
            add_workout_to_user_file(d, outfolder) # add workout to appropriate user file
            uid = d["user_id"]
            users.append(User())
            i += 1
            if (i == 1000):
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reads data and prints statistics about users and workouts')
    parser.add_argument('--infile', type=str, help='one big gzipped file containing all workouts for all users', dest='infile')
    parser.add_argument('--outfolder', type=str, help='path to folder where all user data will be written', dest='outfolder')
    #parser.add_argument('--data', type=str, help='data folder', dest='data_folder')
    #parser.add_argument('--verbose', action='store_true', help='verbose output (default: False)', default=False, dest='verbose')
    #parser.add_argument('--profile', action='store_true', help='profile output (default: False)', default=False, dest='profile')
    #parser.add_argument('--short', action='store_true', help='profile output (default: False)', default=False, dest='short')
    args = parser.parse_args()
    if (args.infile is not None and args.outfolder is not None):
        t1 = time.time()
        #get_user_stats(args.data_folder)
        separate_users(args.infile, args.outfolder)
        t2 = time.time()
        print "Time taken = " + str(t2 - t1)
    else:
        parser.print_usage();
