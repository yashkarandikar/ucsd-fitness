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

class SqlToJsonParserStats(object):
    def __init__(self, lines_parsed, workouts, workouts_invalid, workouts_without_data, workouts_without_info):
        self.lines_parsed = lines_parsed
        self.workouts = workouts
        self.workouts_invalid = workouts_invalid
        self.workouts_without_data = workouts_without_data
        self.workouts_without_info = workouts_without_info

class Workout(object):
    def __init__(self, w_str, status):
        self.w_str = w_str
        self.status = status

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
        #print "Workout " + str(workout_id) + " invalid.."
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
    if (w_str is not None):
        fd.write(w_str + "\n")
        return True
    else:
        return False

def gen_outfile_from_pid(pid, outfile_base):
    fName, fExt = os.path.splitext(outfile_base)
    return fName + "." + str(pid) + fExt

def handle_html(workout_id, html, outfile_base, workouts_queue):
    try:
        [w_dict, w_str, w_md5, status] = parse_html(workout_id, html)
        workouts_queue.put(Workout(w_str, status))
        return status
    except Exception as e:
        pid = mp.current_process().pid
        print "Error on process " + str(pid) + ":" + e.message
        return None

def handle_workout_writes(workouts_queue, outfile):
    fd = gzip.open(outfile, "w")
    n = 0
    while True:
        w = workouts_queue.get()
        if (w == "done"):
            break
        write_workout(None, w.w_str, None, fd)
        n = n + 1
        if (n % 1000 == 0):
            print "Writer: Done writing %d workouts to disk.. Current approximate queue size = %d" % (n, workouts_queue.qsize())
    fd.close()

class SqlToJsonParser(object):

    def __init__(self, infile = "", outfile = "", verbose=False, nprocesses=1, maxPendingWritesQ = 10000, maxPendingResultsQ = 10000):
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
        #self.max_workouts = max_workouts
        self.verbose = verbose
        #self.workout_hashes = {}
        self.maxPendingWritesQ = maxPendingWritesQ
        self.maxPendingResultsQ = maxPendingResultsQ

    def print_stats(self):
        print "Main: # lines parsed = ", self.lines_parsed
        #print "# duplicate workouts = ", self.duplicate_workouts
        print "Main: # workouts successfully extracted = ", self.workouts
        print "Main: # workouts not valid = ", self.workouts_invalid
        print "Main: # workouts without trace data = ", self.workouts_without_data
        print "Main: # workouts without info box = ", self.workouts_without_info
        #print "# users = ", self.users

    def update_stats(self, status):
        if (status.valid):
            self.workouts += 1
            if (not status.has_info):
                self.workouts_without_info += 1
            if (not status.has_trace):
                self.workouts_without_data += 1
        else:
            self.workouts_invalid += 1


    def get_stats(self):
        return SqlToJsonParserStats(lines_parsed = self.lines_parsed,
                                    workouts = self.workouts,
                                    workouts_invalid = self.workouts_invalid,
                                    workouts_without_data = self.workouts_without_data,
                                    workouts_without_info = self.workouts_without_info
                                    )

    def clear_pending_results(self, jobs):
        for j in jobs:
            j.wait()
            assert(j.successful())
            status = j.get()
            self.update_stats(status)
        
    def run(self):
        start_time = time.time()

        print "Main: Max pending writes queue size = ", self.maxPendingWritesQ
        print "Main: Max pending results queue size = ", self.maxPendingResultsQ
        
        # check if input file exists
        infile = self.infile
        if (self.infile == ""):
            raise Exception("Input file not supplied")
        if (not os.path.isfile(infile)):
            print "File not found.."
            exit(0)

        # create workers
        pool = mp.Pool(processes = self.np - 1) # 1 will be the writer queue
        workouts_queue = mp.Manager().Queue(maxsize = self.maxPendingWritesQ)
        writer = mp.Process(target = handle_workout_writes, args=(workouts_queue, self.outfile,))
        writer.start()
        jobs = []
        print "Main: Created pool of " + str(self.np) + " processes.."

        w_count = 0

        print "Main: Reading file " + infile
        # now read input file
        with gzip.open(infile) as f:
            workout_id = ""
            html = ""
            n_lines = 0
            for line in f:
                n_lines += 1
                if ("INSERT INTO `EndoMondoWorkouts` VALUES" in line):  # ignore all other lines
                    self.lines_parsed += 1
                    start = line.find("(")
                    n_html = 0
                    while(start != -1):
                        #p2 = line.find("<!DOCTYPE html", start + 1)    # Fix to important bug which caused it to loop around
                        #workout_id = line[start + 1 : p2 - 2]
                        #start = p2
                        p2 = line.find(",", start + 1)
                        workout_id = line[start + 1 : p2]
                        start = p2 + 2
                        end = line.find("</html>", start + 1)
                        html = line[start : end + len("</html>")]
                        n_html += 1
                        jobs.append(pool.apply_async(handle_html, args = (workout_id[:], html[:], self.outfile, workouts_queue)))
                        start = line.find("(", end + 1)
                        if (len(jobs) > self.maxPendingResultsQ):
                            print "Main: Processing stats of finished jobs.."
                            self.clear_pending_results(jobs)
                            del jobs
                            jobs = []
                            print "Main: Cleared pending job results.. "
                        #print "%d,%d" % (n_lines, n_html)
        
        print "Main: Assigned jobs to threads.. "
        
        # Update stats
        self.clear_pending_results(jobs)
            
        workouts_queue.put("done")  # to signal the writer process
        
        print "Main: Closing pool.."
        pool.close()
        pool.join()

        print "Main: Waiting for writer process to end.."
        writer.join()

        print "Main: Done.."
        end_time = time.time()
        self.print_stats()
        print "Main: Total time taken = ", end_time - start_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reads SQL file and dumps the required data into JSON format, one line per workout')
    parser.add_argument('--infile', type=str, help='.sql.gz file', dest='infile')
    parser.add_argument('--outfile', type=str, help='.gz file', dest='outfile')
    parser.add_argument('--nproc', type=int, help='number of processes to spawn. Note that total number of processes will be nproc + 1', dest='np', default=2)
    parser.add_argument('--maxPendingWritesQ', type=int, help='Maximum size of queue of workouts to be written to disk. If this size is crossed, any put() operation on the queue will be blocked (default : 10000)', dest='maxPendingWritesQ', default=10000)
    parser.add_argument('--maxPendingResultsQ', type=int, help='Maximum number of jobs which are completed but whose results are not processed by the main process.  If this size is crossed, the main process will first process these results and then continue adding more jobs to the pool(default : 10000)', dest='maxPendingResultsQ', default=10000)
    parser.add_argument('--verbose', action='store_true', help='verbose output (default: False)', default=False, dest='verbose')
    parser.add_argument('--profile', action='store_true', help='profile output (default: False)', default=False, dest='profile')
    #parser.add_argument('--short', action='store_true', help='profile output (default: False)', default=False, dest='short')
    args = parser.parse_args()

    #check if all required options are available
    if (args.infile is None or args.outfile is None):
        parser.print_usage()
        sys.exit(0)

    # run, considering all options
    #max_workouts = -1
    #if (args.short):
        #max_workouts = 1000
    s = SqlToJsonParser(args.infile, args.outfile, nprocesses=args.np, maxPendingWritesQ = args.maxPendingWritesQ, maxPendingResultsQ = args.maxPendingResultsQ)
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
