#!/usr/bin/python

import sys, os
import ujson
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath,'..'))
from sql_to_json_parser import SqlToJsonParser
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
    dict_expected = ujson.load(f, precise_float=True)
    f.close()
    assert(dict_obtained == dict_expected)

    # check date-time
    assert(p.extract_date_time(html) == "Apr 21, 2014 10:13 AM")


def json_to_dict(infile):
    f = open(infile)
    d = ujson.load(f, precise_float=True)
    f.close()
    return d

def test_add_workout_to_user():
    outfolder = "/tmp/fitness"
    if (os.path.isdir(outfolder)):
        shutil.rmtree(outfolder)
    os.mkdir(outfolder)
    cwd = os.path.dirname(os.path.abspath(__file__))
    j = json_to_dict(os.path.join(cwd, "./data/2_expected_1.json"))
    user_id = "15549606"
    info_dict = j['workouts'][0]
    data_dict = {}
    data_dict['data'] = info_dict['data']
    del info_dict['data']
    
    # test writing first workout
    p = SqlToJsonParser()
    p.add_workout_to_user(user_id, data_dict, info_dict, outfolder)
    assert(os.path.isfile("/tmp/fitness/15549606.json"))
    assert(json_to_dict("/tmp/fitness/15549606.json") == json_to_dict(os.path.join(cwd, "./data/2_expected_1.json")))

    # test writing 2nd workout
    p.add_workout_to_user(user_id, data_dict, info_dict, outfolder)
    assert(os.path.isfile("/tmp/fitness/15549606.json"))
    assert(json_to_dict("/tmp/fitness/15549606.json") == json_to_dict(os.path.join(cwd, "./data/2_expected_2.json")))


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
    p.parse_html("12345", html, outfolder)
    assert(os.path.isfile("/tmp/fitness/15549606.json"))
    assert(json_to_dict("/tmp/fitness/15549606.json") == json_to_dict(os.path.join(cwd, "./data/2_expected_1.json")))


if __name__ == "__main__":
    test_extract()
    test_add_workout_to_user()
    test_parse_html()
