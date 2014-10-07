#!/usr/bin/python

import sys
import os.path
import json
import re

from HTMLParser import HTMLParser

class InfoBoxHTMLParser(HTMLParser):
    
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
    
    def handle_endtag(self, tag):
        #print "Encountered an end tag :", tag
        pass
    
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


def parse_json(json_data):
    j = json.loads(json_data)
    j2 = {}
    j2['999999'] = j['data']
    json.dump(j2, open("json_dump.txt","w"))


def parse_user_event(html, outfile):
    # html string contains ONE user id, all trace data and info box data
    # this data is extracted and appended to outfile

    # extract user id
    start = html.find("/workouts/user/")
    if (start == -1):
        raise Exception("illegal input to this parse_user_event()..")
    start = start + len("/workouts/user/")
    end = html.find("\\\"", start)
    user_id = int(html[start:end])
    print "user id = " + str(user_id)

    # extract info from 'data'
    start = html.find("\"data\\\"")
    end = html.find("]", start)
    json_string = "{" + html[start : end + 1] + "}"
    json_string = re.sub(r'\\n',r'',json_string)
    json_string = re.sub(r'\\"',r'"',json_string)
    #f = open("json_data.txt","w")
    #f.write(json_data)
    #f.close()
    #parse_json(json_data)

    # extract info box - hydration, wind etc.
    # look for the string "<div class="tab-panel">" and then find the end tag
    start = html.find("<div class=\\\"tab-panel\\\">")
    #print "found tab panel at index " + str(start)
    start = html.find("<ul class=\\\"summary clearfix\\\">", start + 1)
    #print "found ul element at " + str(start)
    end = html.find("</ul>", start + 1)
    info_box_string = html[start : end+5]
    info_box_string = re.sub(r'\\n',r'', info_box_string)
    info_parser = InfoBoxHTMLParser()
    info_parser.feed(info_box_string)
    info_box = info_parser.get_info() # dictionary of all information in info box
    print info_box
    #f = open("info_box.txt",'w')
    #f.write(info_box_string)
    #f.close()
        

def parse_html(workout_id, html):
    # Eventually this should extract relevant data

    user_start = html.find("/workouts/user")
    if (user_start == -1):
        raise Exception("No user found..")
    while(user_start < len(html)):
        user_end = html.find("/workouts/user", user_start + 1)
        if (user_end == -1):
            user_end = len(html)
        user_string = html[user_start:user_end]
        user_start = user_end
        parse_user_event(user_string, "dummy.txt")
        #f = open("user_data.txt",'w')
        #f.write(user_string)
        #f.close()
        break   # added for now, to restrict to one user

    #while (True):
    #index = html.find("/workouts/user", start)
    #if (index == -1):
        #break
    #users.append(index)
    #print "found user at index " + str(index)
    #start = index + 1
    #start = index + 1


def parse_sql_file(infile):
    if (not os.path.isfile(infile)):
        print "File not found.."
        exit(0)

    print "Reading file " + infile

    with open(infile) as f:
        workout_id = ""
        html = ""
        n_records = 0
        for line in f:
            if ("INSERT INTO `EndoMondoWorkouts` VALUES" in line):  # ignore all other lines
                record = line.split("INSERT INTO `EndoMondoWorkouts` VALUES ")[1].strip()
                comma_index = record.find(',')  # separate workout_id and html string
                workout_id = record[1:comma_index]
                html = record[comma_index + 1:]
                print workout_id
                parse_html(workout_id, html)    # this will extract relevant data

                n_records = n_records + 1
                if (n_records == 1):       # stop after 10 records for testing
                    break;


if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print "Usage: parse_csv.py <file>"
        exit(0)
    infile = sys.argv[1]
    parse_sql_file(infile)
