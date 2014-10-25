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


class SqlToJsonParser(object):

    def __init__(self, sqlfile = "", max_users = -1, max_workouts = -1, verbose=False):
        self.sqlfile = sqlfile
        self.workouts_without_user = 0
        self.duplicate_workouts = 0
        self.lines_parsed = 0
        self.users = 0
        self.workouts = 0
        self.workouts_without_data = 0
        self.workouts_without_info = 0
        self.max_users = max_users
        self.max_workouts = max_workouts
        self.verbose = verbose
        self.workout_hashes = {}

    def add_workout_to_user(self, user_id, workout_id, sport_type, trace_dict, info_dict, outfolder):
        # creates a file for the user (if does not already exist) and adds a workout in JSON format
        folder = os.path.join(outfolder, str(user_id)[:3])
        if (not os.path.isdir(folder)):
            os.mkdir(folder)
        filepath = os.path.join(folder, str(user_id) + ".txt")
        
        # Create ONE dictionary of all information and data of ONE workout done by one user and convert it to a string
        # compute hash, check if its duplicate and add to the dictionary in memory for future comparisons
        uid = int(user_id)
        workout_dict = info_dict    
        #if (data_dict.has_key('data')):
            #workout_dict['data'] = data_dict['data']
        workout_dict.update(trace_dict)
        workout_dict["sport"] = sport_type
        #workout_str = ujson.dumps(workout_dict, double_precision=15)  # convert to string using ujson library
        workout_md5 = hashlib.md5(json.dumps(workout_dict)).hexdigest()  # exclude workout_id when calculating md5, otherwise duplicates will not get detected
        workout_dict["workout_id"] = workout_id
        workout_str = json.dumps(workout_dict)  # convert to string using json library
        if (not os.path.isfile(filepath)):
            self.users += 1
            self.workout_hashes[uid] = [workout_md5]
        else:
            # check for duplicates
            if (workout_md5 in self.workout_hashes[uid]):
                self.duplicate_workouts += 1
                return False
            self.workout_hashes[uid].append(workout_md5)

        # add to user's file
        with open(filepath, 'a') as f:
            f.write(workout_str + "\n")
        self.workouts += 1
        return True

    def extract_user_id(self, html):
        # exitract user id
        start = html.find("/workouts/user/")
        if (start == -1):
            raise Exception("illegal input to this parse_user_event()..")
        start = start + len("/workouts/user/")
        end = html.find("\\\"", start)
        user_id = int(html[start:end])
        if (html.find("/workouts/user/", start) != -1):
            raise Exception("Multiple user ids in one html doc")
        return user_id

    def reformat_trace_data(self, ori_data):
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
                    raise Exception("Unknown key %s found in trace data" %(k))

        # check if all lists are of same length and remove any keys with all N's
        l = len(new_data["lng"])
        for k in counts:
            assert(l == len(new_data[k]))
            if (counts[k] == 0):
                #print "key %s entirely absent.. so removing it.." %(k)
                del new_data[k]

        return new_data

    def extract_trace_data(self, html):
        # extract info from 'data' - these are the traces (gps, heart-rate, pace)
        trace_data = {}
        json_string = ""
        start = html.find("\"data\\\"")
        if (start != -1):
            end = html.find("]", start)
            json_string = "{" + html[start : end + 1] + "}"
            json_string = re.sub(r'\\n',r'',json_string)
            json_string = re.sub(r'\\"',r'"',json_string)
            #trace_data = ujson.loads(json_string, precise_float=True)
            trace_data = json.loads(json_string)
            trace_data = self.reformat_trace_data(trace_data)
        else:
            self.workouts_without_data += 1
        if (html.find("\"data\\\"", start + 1) != -1):
            raise Exception("Multiple 'data' elements in one html doc")
        return trace_data 

    def extract_info_box(self, html):
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
            self.workouts_without_info += 1
        if (html.find("<div class=\\\"tab-panel\\\">", start + 1) != -1):
            raise Exception("Multiple tab-panels in one html doc")
        return info_data


    def extract_date_time(self, html):
        # extract date and time
        start = html.find("<div class=\\\"date-time\\\">")
        date_time_string = None
        if (start != -1):
            #print "Found date-time.."
            end = html.find("</div>", start + 1)
            date_time_string = html[start + len("<div class=\\\"date-time\\\">") : end]
        if (html.find("<div class=\\\"date-time\\\">", start + 1) != -1):
            raise Exception("Multiple date-time elements in one html")
        return date_time_string

    def extract_sport_type(self, html):
        start  = html.find("<div class=\\\"sport-name\\\">")
        if (start != -1):
            end = html.find("</div>", start + 1)
            type_string = html[start + len("<div class=\\\"sport-name\\\">") : end]
        if (html.find("<div class=\\\"sport-name\\\">", start + 1) != -1):
            raise Exception("Multiple sport-name elements in one html")
        return type_string


    def parse_user_event(self, workout_id, html, outfolder):
        # html string contains ONE user id, all trace data and info box data
        
        # extract user id
        user_id = self.extract_user_id(html)

        # extract sport type - cycling, walking etc
        sport_type = self.extract_sport_type(html)
        
        # extract info from 'data' - these are the traces (gps, heart-rate, pace)
        trace_data = self.extract_trace_data(html)
        
        # extract info box - hydration, wind etc.
        info_data = self.extract_info_box(html)
        
        # extract date and time string
        date_time_string = self.extract_date_time(html)
        if (date_time_string is not None):
            info_data['date-time'] = date_time_string
                
        # add to user's file
        self.add_workout_to_user(user_id, workout_id, sport_type, trace_data, info_data, outfolder)

    def parse_html(self, workout_id, html, outfolder):
        #if (self.verbose):
            #print "Processing workout " + workout_id
        start = html.find("/workouts/user")
        if (start == -1):
            #if (self.verbose):
                #print "\tNo user found..."
            self.workouts_without_user += 1
            return
        if (html.find("/workouts/user", start + 1) != -1):
            raise Exception("Multiple users found in one html string..")
        self.parse_user_event(workout_id, html, outfolder)

    def print_stats(self):
        print "# lines parsed = ", self.lines_parsed
        print "# duplicate workouts = ", self.duplicate_workouts
        print "# workouts_without_user = ", self.workouts_without_user
        print "# workouts successfully extracted = ", self.workouts
        print "# users = ", self.users
        print "# workouts without trace data = ", self.workouts_without_data
        print "# workouts without info box = ", self.workouts_without_info

    def done(self):
        if (self.max_users > 0 and self.users >= self.max_users):
            return True
        if (self.max_workouts > 0 and self.workouts >= self.max_workouts):
            return True
        return False

    def run(self):
        start_time = time.time()

        infile = self.sqlfile
        if (self.sqlfile == ""):
            raise Exception("Input file not supplied")
        if (not os.path.isfile(infile)):
            print "File not found.."
            exit(0)

        print "Reading file " + infile

        # create output folder name
        infile_name, infile_ext = os.path.splitext(infile);
        infile_name = os.path.basename(infile_name)
        outfolder_base = infile_name.split(".")[0]
        if (outfolder_base == ""):
            outfolder_base = "temp"
        outfolder = os.path.join(os.path.dirname(__file__),"..","data",outfolder_base)
        if (os.path.isdir(outfolder)):
            print outfolder
            shutil.rmtree(outfolder)
            print "Removed existing folder " + outfolder
        os.mkdir(outfolder)
        print "Created folder " + outfolder

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
                            self.parse_html(workout_id, html, outfolder)
                            start = line.find("(", end + 1)
                            #f2 = open("test_html.html",'w')
                            #f2.write(html)
                            #f2.close()

        # tar the folder
        #print "Compressing to tar.gz"
        #gz_path = outfolder + ".tar.gz"
        #if (os.path.isfile(gz_path)):
        #    print "Deleting existing tar.gz file.."
        #    os.remove(gz_path)
        #with tarfile.open(gz_path, "w:gz") as tar:
        #    tar.add(outfolder, arcname = os.path.basename(outfolder))
        
        end_time = time.time()
        self.print_stats()
        print "Total time taken = ", end_time - start_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reads SQL file and dumps the required data into JSON format')
    parser.add_argument('--infile', type=str, help='.SQL file', dest='infile')
    parser.add_argument('--verbose', action='store_true', help='verbose output (default: False)', default=False, dest='verbose')
    parser.add_argument('--profile', action='store_true', help='profile output (default: False)', default=False, dest='profile')
    parser.add_argument('--short', action='store_true', help='profile output (default: False)', default=False, dest='short')
    args = parser.parse_args()
    if (args.infile is not None):
        max_workouts = -1
        if (args.short):
            max_workouts = 1000
        s = SqlToJsonParser(args.infile, max_workouts=max_workouts)
        if (args.profile):
            print "Running in profiling mode.."
            pr = cProfile.Profile()
            pr.enable()
            s.run()
            pr.disable()
            s = StringIO.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print s.getvalue()
        else:
            s.run()
    else:
        parser.print_usage()
