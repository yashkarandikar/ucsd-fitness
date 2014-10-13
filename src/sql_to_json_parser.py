#!/usr/bin/python

import sys
import os.path
import ujson
import re
import shutil
from HTMLParser import HTMLParser
import time
import argparse
import profile
import tarfile

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

    def add_workout_to_user(self, user_id, data_dict, info_dict, outfolder):
        # creates a file for the user (if does not already exist) and adds a workout in JSON format
        filepath = os.path.join(outfolder, str(user_id) + ".json")
        
        # Create ONE dictionary of all information and data of ONE workout done by one user
        workout_dict = info_dict    
        if (data_dict.has_key('data')):
            workout_dict['data'] = data_dict['data']
        
        # add to user's file
        duplicate = False
        if (os.path.isfile(filepath)):  # if user's file already exists, read the full thin and add the new workout
            # read all workouts from user's file
            f = open(filepath)
            j = ujson.load(f, precise_float=True)
            f.close()
            
            # check if this workout is duplicate
            workouts = j['workouts']
            for w in workouts:
                if (str(workout_dict) == str(w)):
                    duplicate = True
                    self.duplicate_workouts += 1
                    return
            
            j['workouts'].append(workout_dict)  # add a dict to the list of dicts
            #if (self.verbose):
            #    print "User "+ str(user_id) + " has more than 1 workout.."
        else:           # else just create a new workout
            self.users += 1
            j = {}
            j['id'] = str(user_id)
            j['workouts'] = [workout_dict]
       
        self.workouts += 1

        # write back everything to the file
        #if (self.verbose):
            #print "Writing to " + filepath
        f = open(filepath, "w") # not a very efficient way of adding something to a file, but okay for now
        ujson.dump(j, f, double_precision=15)
        f.close()

    def extract_user_id(self, html):
        # extract user id
        start = html.find("/workouts/user/")
        if (start == -1):
            raise Exception("illegal input to this parse_user_event()..")
        start = start + len("/workouts/user/")
        end = html.find("\\\"", start)
        user_id = int(html[start:end])
        if (html.find("/workouts/user/", start) != -1):
            raise Exception("Multiple user ids in one html doc")
        return user_id
        #print "user id = " + str(user_id)


    def extract_trace_data(self, html):
        # extract info from 'data' - these are the traces (gps, heart-rate, pace)
        json_data = {}
        json_string = ""
        start = html.find("\"data\\\"")
        if (start != -1):
            end = html.find("]", start)
            json_string = "{" + html[start : end + 1] + "}"
            json_string = re.sub(r'\\n',r'',json_string)
            json_string = re.sub(r'\\"',r'"',json_string)
            json_data = ujson.loads(json_string, precise_float=True)
        else:
            self.workouts_without_data += 1
        if (html.find("\"data\\\"", start + 1) != -1):
            raise Exception("Multiple 'data' elements in one html doc")
        return json_data

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

    def parse_user_event(self, html, outfolder):
        # html string contains ONE user id, all trace data and info box data
        
        # extract user id
        user_id = self.extract_user_id(html)
        
        # extract info from 'data' - these are the traces (gps, heart-rate, pace)
        trace_data = self.extract_trace_data(html)
        
        # extract info box - hydration, wind etc.
        info_data = self.extract_info_box(html)
        
        # extract date and time string
        date_time_string = self.extract_date_time(html)
        if (date_time_string is not None):
            info_data['date-time'] = date_time_string
                
        # add to user's file
        self.add_workout_to_user(user_id, trace_data, info_data, outfolder)

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
        self.parse_user_event(html, outfolder)

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
        outfolder = os.path.join(os.path.dirname(__file__),"..","data",infile_name)
        if (os.path.isdir(outfolder)):
            shutil.rmtree(outfolder)
            print "Removed existing folder " + outfolder
        os.mkdir(outfolder)
        print "Created folder " + outfolder

        # now read input file
        with open(infile) as f:
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
        with tarfile.open(outfolder + ".tar.gz", "w:gz") as tar:
            tar.add(outfolder, arcname = os.path.basename(outfolder))

        
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
