#!/usr/bin/python

import json
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath,'..'))
from sql_to_json_parser import SqlToJsonParser

def test_extract():
    # read test data file
    cwd = os.path.dirname(os.path.abspath(__file__))
    infile = os.path.join(cwd, "./data","1.html")
    f = open(infile)
    html = f.read()
    f.close()
    
    p = SqlToJsonParser()
    
    # check user id
    assert(str(p.extract_user_id(html)) == "15549606")   # check user id

    # check info box
    dict_obtained = p.extract_info_box(html)
    dict_expected = {'Distance' : '2.35 mi', 'Duration' : '24m:03s', 'Avg. Speed' : '10:13 min/mi', 'Max. Speed' : '7:01 min/mi', 'Calories' : '326 kcal', 'Hydration' : '0.34L', 'Min. Altitude' : '456 ft', 'Max. Altitude' : '577 ft', 'Total Ascent' : '417 ft', 'Total Descent' : '486 ft', 'Weather' : 'Sunny'}
    assert(dict_obtained == dict_expected)

    # check trace data
    dict_obtained = p.extract_trace_data(html)
    f = open(os.path.join(cwd, "./data","1_trace.json"))
    dict_expected = json.load(f)
    f.close()
    assert(dict_obtained == dict_expected)

    # check date-time
    assert(p.extract_date_time(html) == "Apr 21, 2014 10:13 AM")


if __name__ == "__main__":
    test_extract()
