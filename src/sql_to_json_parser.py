#!/usr/bin/python

import sys
import os.path
#import ujson
import simplejson as json
import re
import shutil
from HTMLParser import HTMLParser
import time
import argparse
import profile
import tarfile
import hashlib
import gzip
import cProfile
import StringIO
import pstats
import multiprocessing as mp
import utils

class InfoBoxHTMLParser(HTMLParser):
    # this class will take a html string corresponding to the info box (hydration, max altitude etc.) and return a dictionary after extracting the data
    
    def __init__(self):
        #super(InfoBoxHTMLParser, self).__init__()
        HTMLParser.__init__(self)
        self.info = {}
        self.waiting_for_label = False
        self.waiting_for_value = False
    
    def handle_starttag(self, tag, attrs):
        if (tag == "span"):
            if (('class', '\\"label\\"') in attrs):
                self.waiting_for_label = True
            elif (('class', '\\"value\\"') in attrs):
                self.waiting_for_value = True
    
    def handle_data(self, data):
        if (self.waiting_for_label):
            self.label = data
            self.waiting_for_label = False
        elif (self.waiting_for_value):
            self.info[self.label] = data
            self.waiting_for_value = False
    
    def get_info(self):
        return self.info

class WorkoutStatus(object):
    def __init__(self, has_info = True, has_trace = True, valid = True):
        self.has_info = has_info
        self.has_trace = has_trace
        self.valid = valid

def compute_md5( workout_dict):
    return hashlib.md5(str(workout_dict)).hexdigest()

def create_workout_dict( user_id, workout_id, sport_type, trace_dict, info_dict):
    # Create ONE dictionary of all information and data of ONE workout done by one user and convert it to a string
    #print "in create workout dict"
    uid = int(user_id)
    workout_dict = info_dict
    #print "just before update"
    #print trace_dict
    if (trace_dict is not None):
        workout_dict.update(trace_dict)
    #print "just after update"
    workout_dict["sport"] = sport_type
    workout_dict["user_id"] = str(user_id)
    #workout_md5 = hashlib.md5(json.dumps(workout_dict)).hexdigest()  # exclude workout_id when calculating md5, otherwise duplicates will not get detected
    #workout_md5 = self.compute_md5(workout_dict) # exclude workout_id when calculating md5, otherwise duplicates will not get detected
    workout_dict["workout_id"] = workout_id
    workout_str = json.dumps(workout_dict)  # convert to string using json library
    #print "returning from create workout dict"
    has_info = (info_dict is not None)
    has_trace = (trace_dict is not None)
    return [workout_dict, workout_str, None, WorkoutStatus(has_info, has_trace, valid = True)]

def extract_user_id( html):
    # exitract user id
    start = html.find("/workouts/user/")
    if (start == -1):
        print "ILLEGAL INPUT"
        raise Exception("illegal input to this parse_user_event()..")
    start = start + len("/workouts/user/")
    end = html.find("\\\"", start)
    user_id = int(html[start:end])
    if (html.find("/workouts/user/", start) != -1):
        print "MULTIPLE USER IDS.."
        raise Exception("Multiple user ids in one html doc")
    return user_id

def reformat_trace_data( ori_data):
    """
    ori_data is  trace data for ONE user
    """
    records = ori_data['data']
    new_data = {"lng":[], "lat":[], "alt":[], "duration":[], "distance":[], "speed" : [], "pace" : [], "hr" : [], "cad":[]}
    for r in records:
        # r itself is a dict of the form {"lat": 50.223, "lng": 19.247, "values": {"duration": 19352, "distance": 0.0, "alt": 1040, "speed": 0.8}}
        # remove the nested dictionary and make it flat
        if (r.has_key("values")):
            values = r['values']
            del r['values']
            r.update(values)
        
        # initialize a counts dictionary, which keeps a count of non-N entries, to allow fast deletion of lists which are just N's
        counts = {}
        for k in new_data:
            counts[k] = 0

        for k in new_data:
            if (r.has_key(k)):
                new_data[k].append(round(r[k], 6))
                counts[k] += 1
            else:
                new_data[k].append("N")

        for k in r:
            if (k not in new_data.keys()):
                print "UNKNOWN KEY FOUND IN TRACE DATA"
                raise Exception("Unknown key %s found in trace data" %(k))

    # check if all lists are of same length and remove any keys with all N's
    l = len(new_data["lng"])
    for k in counts:
        assert(l == len(new_data[k]))
        if (counts[k] == 0):
            #print "key %s entirely absent.. so removing it.." %(k)
            del new_data[k]

    return new_data

def extract_trace_data( html):
    # extract info from 'data' - these are the traces (gps, heart-rate, pace)
    trace_data = {}
    json_string = ""
    start = html.find("\"data\\\"")
    if (start != -1):
        end = html.find("]", start)
        json_string = "{" + html[start : end + 1] + "}"
        json_string = re.sub(r'\\n',r'',json_string)
        json_string = re.sub(r'\\"',r'"',json_string)
        trace_data = json.loads(json_string)
        trace_data = reformat_trace_data(trace_data)
    else:
        trace_data = None
        #self.workouts_without_data += 1
    if (html.find("\"data\\\"", start + 1) != -1):
        print "MULTIPLE DATA ELEMENTS IN ONE HTML DOC"
        raise Exception("Multiple 'data' elements in one html doc")
    return trace_data 

def extract_info_box( html):
    # extract info box - hydration, wind etc.
    # look for the string "<div class="tab-panel">" and then find the end tag
    info_data = {}
    start = html.find("<div class=\\\"tab-panel\\\">")
    if (start != -1):
        start = html.find("<ul class=\\\"summary clearfix\\\">", start + 1)
        end = html.find("</ul>", start + 1)
        info_box_string = html[start : end + len("</ul>")]
        info_box_string = re.sub(r'\\n',r'', info_box_string)
        info_parser = InfoBoxHTMLParser()
        info_parser.feed(info_box_string)
        info_data = info_parser.get_info() # dictionary of all information in info box
    else:
        info_data = None
        #self.workouts_without_info += 1
    if (html.find("<div class=\\\"tab-panel\\\">", start + 1) != -1):
        print "MULTIPLE TAB PANELS IN ONE HTML DOC"
        raise Exception("Multiple tab-panels in one html doc")
    return info_data

def extract_date_time( html):
    # extract date and time
    start = html.find("<div class=\\\"date-time\\\">")
    date_time_string = None
    if (start != -1):
        #print "Found date-time.."
        end = html.find("</div>", start + 1)
        date_time_string = html[start + len("<div class=\\\"date-time\\\">") : end]
    if (html.find("<div class=\\\"date-time\\\">", start + 1) != -1):
        print "MULTIPLE DATE-TIME ELEMENTS IN ONE HTML"
        raise Exception("Multiple date-time elements in one html")
    return date_time_string

def extract_sport_type( html):
    start  = html.find("<div class=\\\"sport-name\\\">")
    if (start != -1):
        end = html.find("</div>", start + 1)
        type_string = html[start + len("<div class=\\\"sport-name\\\">") : end]
    if (html.find("<div class=\\\"sport-name\\\">", start + 1) != -1):
        print "MULTIPLE SPORT-NAME ELEMENTS IN ONE HTML"
        raise Exception("Multiple sport-name elements in one html")
    return type_string

def parse_user_event( workout_id, html):
    #print "\t In parse_user_event"
    # html string contains ONE user id, all trace data and info box data
    
    # extract user id
    user_id = extract_user_id(html)
    #print "\tdone user id.."

    # extract sport type - cycling, walking etc
    sport_type = extract_sport_type(html)
    #print "\tdone sport type.."
    
    # extract info from 'data' - these are the traces (gps, heart-rate, pace)
    trace_dict = extract_trace_data(html)
    #print "\tdone trace dict.."
    
    # extract info box - hydration, wind etc.
    info_dict = extract_info_box(html)
    #print "\tdone info dict.."
    
    # extract date and time string
    date_time_string = extract_date_time(html)
    if (date_time_string is not None):
        info_dict['date-time'] = date_time_string
    #print "\tdone date time"
       
    # combine everything into one dictionary
    return create_workout_dict(user_id, workout_id, sport_type, trace_dict, info_dict)

def parse_html( workout_id, html):
    #print "\tIn parse_html.."
    start = html.find("/workouts/user")
    if (start == -1):
        #self.workouts_invalid += 1
        return [None, None, None, WorkoutStatus(valid = False)]
    if (html.find("/workouts/user", start + 1) != -1):
        print "MULTIPLE USERS FOUND IN ONE HTML STRING"
        raise Exception("Multiple users found in one html string..")
    [w_dict, w_str, w_md5, status] = parse_user_event(workout_id, html)
    return [w_dict, w_str, w_md5, status]

def write_workout( w_dict, w_str, w_md5, fd):
    """
    if (w_dict is not None):
        uid = int(w_dict["user_id"])
        if (self.workout_hashes.has_key(uid)):    # existing user
            if (w_md5 in self.workout_hashes[uid]):     # duplicate workout for that user
                self.duplicate_workouts += 1
                return False
            else:                                   # new workout for existing user
                fd.write(w_str + "\n")
                self.workouts += 1
                self.workout_hashes[uid].append(w_md5)
        else:                                   # new user , new workout
            fd.write(w_str + "\n")
            self.users += 1
            self.workouts += 1
            self.workout_hashes[uid] = [w_md5]
    """
    if (w_dict is not None):
        fd.write(w_str + "\n")
        return True
    else:
        return False

def gen_outfile_from_pid(pid, outfile_base):
    fName, fExt = os.path.splitext(outfile_base)
    return fName + "." + str(pid) + fExt

def handle_html(workout_id, html, outfile_base):
    pid = mp.current_process().pid
    outfile = gen_outfile_from_pid(pid, outfile_base)
    out_fd = gzip.open(outfile, "a")
    [w_dict, w_str, w_md5, status] = parse_html(workout_id, html)
    write_workout(w_dict, w_str, w_md5, out_fd)
    out_fd.close()
    return [pid, status]

class SqlToJsonParser(object):

    def __init__(self, infile = "", outfile = "", max_workouts = -1, verbose=False, nprocesses=1):
        self.np = nprocesses
        self.infile = infile
        self.outfile = outfile
        self.workouts_invalid = 0
        #self.duplicate_workouts = 0
        self.lines_parsed = 0
        #self.users = 0
        self.workouts = 0
        self.workouts_without_data = 0
        self.workouts_without_info = 0
        self.max_workouts = max_workouts
        self.verbose = verbose
        self.workout_hashes = {}

    def print_stats(self):
        print "# lines parsed = ", self.lines_parsed
        #print "# duplicate workouts = ", self.duplicate_workouts
        print "# workouts successfully extracted = ", self.workouts
        print "# workouts not valid = ", self.workouts_invalid
        print "# workouts without trace data = ", self.workouts_without_data
        print "# workouts without info box = ", self.workouts_without_info
        #print "# users = ", self.users

    def done(self):
        #if (self.max_users > 0 and self.users >= self.max_users):
            #return True
        if (self.max_workouts > 0 and self.workouts >= self.max_workouts):
            return True
        return False
    
    def run(self):
        start_time = time.time()
        
        pool = mp.Pool(processes = self.np)

        infile = self.infile
        if (self.infile == ""):
            raise Exception("Input file not supplied")
        if (not os.path.isfile(infile)):
            print "File not found.."
            exit(0)

        print "Reading file " + infile
        
        jobs = []

        # now read input file
        with gzip.open(infile) as f:
            workout_id = ""
            html = ""
            n_records = 0
            for line in f:
                if (self.done()):
                    break
                if ("INSERT INTO `EndoMondoWorkouts` VALUES" in line):  # ignore all other lines
                    self.lines_parsed += 1
                    start = line.find("(")
                    while(start != -1 and (not self.done())):
                        p2 = line.find("<!DOCTYPE html", start + 1)
                        workout_id = line[start + 1 : p2 - 2]
                        start = p2
                        end = line.find("</html>", start + 1)
                        html = line[start : end + len("</html>")]
                        #if (len(jobs) == 28):
                        #    print "This job is going to fail : "
                        #    print "Workout ID = " + str(workout_id)
                        #    jobs.append(pool.apply_async(handle_html, args = (workout_id[:], html[:], self.outfile)))
                        #    jobs[28].wait()
                        #    assert(jobs[28].successful())
                        jobs.append(pool.apply_async(handle_html, args = (workout_id[:], html[:], self.outfile)))
                        #pool.apply_async(test_fn)
                        start = line.find("(", end + 1)

        pids = set()
        print "Waiting for processes to finish"
        for j in range(0, len(jobs)):
            jobs[j].wait()
            if (not jobs[j].successful()):
                print "Job " + str(j) + " not succesful.."
            assert(jobs[j].successful())
            [pid, status] = jobs[j].get()
            pids.add(pid)
            if (status.valid):
                self.workouts += 1
                if (not status.has_info):
                    self.workouts_without_info += 1
                if (not status.has_trace):
                    self.workouts_without_data += 1
            else:
                self.workouts_invalid += 1
            
        pool.close()
        pool.join()

        # combine all gz files
        print "Combining all gzipped files into one.."
        gzfiles = []
        for pid in pids:
            gzfiles.append(gen_outfile_from_pid(pid, self.outfile))
        print gzfiles
        utils.combine_gzip_files(gzfiles, self.outfile)
        print "Deleting all part gzipped files"
        for gzf in gzfiles:
            os.remove(gzf)

        print "Done"
        end_time = time.time()
        self.print_stats()
        print "Total time taken = ", end_time - start_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reads SQL file and dumps the required data into JSON format, one line per workout')
    parser.add_argument('--infile', type=str, help='.sql.gz file', dest='infile')
    parser.add_argument('--outfile', type=str, help='.gz file', dest='outfile')
    parser.add_argument('--verbose', action='store_true', help='verbose output (default: False)', default=False, dest='verbose')
    parser.add_argument('--profile', action='store_true', help='profile output (default: False)', default=False, dest='profile')
    parser.add_argument('--short', action='store_true', help='profile output (default: False)', default=False, dest='short')
    args = parser.parse_args()

    #check if all required options are available
    if (args.infile is None or args.outfile is None):
        parser.print_usage()
        sys.exit(0)

    # run, considering all options
    max_workouts = -1
    if (args.short):
        max_workouts = 1000
    s = SqlToJsonParser(args.infile, args.outfile, max_workouts=max_workouts,nprocesses=3)
    if (args.profile):
        print "Running in profiling mode.."
        pr = cProfile.Profile()
        pr.enable()
        s.run()
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'total'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()
    else:
        s.run()
