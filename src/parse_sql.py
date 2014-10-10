#!/usr/bin/python

import sys
import os.path
import json
import re
import shutil
from HTMLParser import HTMLParser
import collections

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
        #print "Encountered some data  :", data
        if (self.waiting_for_label):
            self.label = data
            self.waiting_for_label = False
        elif (self.waiting_for_value):
            self.info[self.label] = data
            self.waiting_for_value = False
            #print "added key value pair: " + self.label + " = " + self.info[self.label]
    
    def get_info(self):
        return self.info

def add_workout_to_user(user_id, data_dict, info_dict, outfolder):
    # creates a file for the user (if not already exists) and adds a workout in JSON format
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
        j = json.load(f)
        f.close()
        
        # check if this workout is duplicate
        workouts = j['workouts']
        for w in workouts:
            if (str(workout_dict) == str(w)):
                duplicate = True
                print "Detected duplicate !"
                return
        
        j['workouts'].append(workout_dict)  # add a dict to the list of dicts
    else:           # else just create a new workout
        j = {}
        j['id'] = str(user_id)
        j['workouts'] = [workout_dict]
   
    # write back everything to the file
    print "Writing to " + filepath
    f = open(filepath, "w") # not a very efficient way of adding something to a file, but okay for now
    json.dump(j, f)
    f.close()

def parse_user_event(html, outfolder):
    # html string contains ONE user id, all trace data and info box data

    # extract user id
    start = html.find("/workouts/user/")
    if (start == -1):
        raise Exception("illegal input to this parse_user_event()..")
    start = start + len("/workouts/user/")
    end = html.find("\\\"", start)
    user_id = int(html[start:end])
    #print "user id = " + str(user_id)

    # extract info from 'data' - these are the traces (gps, heart-rate, pace)
    json_data = {}
    json_string = ""
    start = html.find("\"data\\\"")
    if (start != -1):
        end = html.find("]", start)
        json_string = "{" + html[start : end + 1] + "}"
        json_string = re.sub(r'\\n',r'',json_string)
        json_string = re.sub(r'\\"',r'"',json_string)
        json_data = json.loads(json_string)

    # extract info box - hydration, wind etc.
    # look for the string "<div class="tab-panel">" and then find the end tag
    info_data = {}
    start = html.find("<div class=\\\"tab-panel\\\">")
    if (start != -1):
        #print "found tab panel at index " + str(start)
        start = html.find("<ul class=\\\"summary clearfix\\\">", start + 1)
        #print "found ul element at " + str(start)
        end = html.find("</ul>", start + 1)
        info_box_string = html[start : end + len("</ul>")]
        info_box_string = re.sub(r'\\n',r'', info_box_string)
        info_parser = InfoBoxHTMLParser()
        info_parser.feed(info_box_string)
        info_data = info_parser.get_info() # dictionary of all information in info box

    # extract date and time
    start = html.find("<div class=\\\"date-time\\\">")
    if (start != -1):
        #print "Found date-time.."
        end = html.find("</div>", start + 1)
        date_time_string = html[start + len("<div class=\\\"date-time\\\">") : end]
        info_data['date-time'] = date_time_string
        
    add_workout_to_user(user_id, json_data, info_data, outfolder)

def parse_html(workout_id, html, outfolder):
    user_start = html.find("/workouts/user")
    if (user_start == -1):
        raise Exception("No user found..")
    while(user_start < len(html)):
        user_end = html.find("/workouts/user", user_start + 1)
        if (user_end == -1):
            user_end = len(html)
        user_string = html[user_start:user_end]
        user_start = user_end
        parse_user_event(user_string, outfolder)
        #break   # for testing, restrict one user

def parse_sql_file(infile):
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
            if ("INSERT INTO `EndoMondoWorkouts` VALUES" in line):  # ignore all other lines
                record = line.split("INSERT INTO `EndoMondoWorkouts` VALUES ")[1].strip()
                comma_index = record.find(',')  # separate workout_id and html string
                workout_id = record[1:comma_index]
                print "workout_id = " + workout_id
                html = record[comma_index + 1:]
                parse_html(workout_id, html, outfolder)    # this will extract relevant data

                n_records = n_records + 1
                #if (n_records == 1):
                    #break


if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print "Usage: parse_csv.py <file>"
        exit(0)
    infile = sys.argv[1]
    parse_sql_file(infile)
