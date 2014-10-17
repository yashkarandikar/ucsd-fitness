#!/usr/bin/python

import sys, os
import simplejson as json
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath,'..'))
from sql_to_json_parser import SqlToJsonParser
from utils import json_to_dicts
import shutil

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
    #dict_expected = ujson.load(f, precise_float=True)
    dict_expected = json.load(f)
    f.close()
    assert(dict_obtained == dict_expected)

    # check date-time
    assert(p.extract_date_time(html) == "Apr 21, 2014 10:13 AM")

def test_add_workout_to_user():
    outfolder = "/tmp/fitness"
    if (os.path.isdir(outfolder)):
        shutil.rmtree(outfolder)
    os.mkdir(outfolder)
    cwd = os.path.dirname(os.path.abspath(__file__))
    dicts = json_to_dicts(os.path.join(cwd, "./data/2_expected_1.txt"))
    user_id = "15549606"
    info_dict = dicts[0]
    data_dict = {}
    data_dict['data'] = info_dict['data']
    del info_dict['data']
    
    # test writing first workout
    p = SqlToJsonParser()
    assert(p.add_workout_to_user(user_id, data_dict, info_dict, outfolder))
    assert(os.path.isfile("/tmp/fitness/15549606.txt"))
    assert(json_to_dicts("/tmp/fitness/15549606.txt") == json_to_dicts(os.path.join(cwd, "./data/2_expected_1.txt")))

    # test writing 2nd workout (duplicate)
    assert (p.add_workout_to_user(user_id, data_dict, info_dict, outfolder) == False)   # False coz we are adding duplicate
    
    # now write a 2nd different workout
    info_dict['Weather'] = "Raining"    # to generate a different workout
    assert (p.add_workout_to_user(user_id, data_dict, info_dict, outfolder))
    assert(os.path.isfile("/tmp/fitness/15549606.txt"))
    assert(json_to_dicts("/tmp/fitness/15549606.txt") == json_to_dicts(os.path.join(cwd, "./data/2_expected_2.txt")))


def test_parse_html():
    outfolder = "/tmp/fitness"
    if (os.path.isdir(outfolder)):
        shutil.rmtree(outfolder)
    os.mkdir(outfolder)
    cwd = os.path.dirname(os.path.abspath(__file__))
    f = open(os.path.join(cwd, "data","1.html"))
    html = f.read()
    f.close()
    p = SqlToJsonParser()
    p.parse_html("15549606", html, outfolder)
    assert(os.path.isfile("/tmp/fitness/15549606.txt"))
    assert(json_to_dicts("/tmp/fitness/15549606.txt") == json_to_dicts(os.path.join(cwd, "./data/2_expected_1.txt")))


if __name__ == "__main__":
    test_extract()
    test_add_workout_to_user()
    test_parse_html()
