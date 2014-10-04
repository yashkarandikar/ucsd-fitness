#!/usr/bin/python

import sys
import os.path

def parse_html(workout_id, html):
    # Eventually this should extract relevant data
    # f = open(workout_id + ".html", 'w')
    # f.write(html)
    # f.close()
    pass


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
                if (n_records == 11):       # stop after 10 records for testing
                    break;


if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print "Usage: parse_csv.py <file>"
        exit(0)
    infile = sys.argv[1]
    parse_sql_file(infile)
