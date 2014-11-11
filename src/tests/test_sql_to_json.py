#!/usr/bin/python

import sys, os
import simplejson as json
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath,'..'))
from sql_to_json_parser import SqlToJsonParser
import sql_to_json_parser 
#from utils import get_workouts
import shutil
import gzip
import utils

def test_extract():
    create_tmp_folder()
    # read test data file
    cwd = os.path.dirname(os.path.abspath(__file__))
    infile = os.path.join(cwd, "./data","1.html")
    f = open(infile)
    html = f.read()
    f.close()
    
    # check user id
    assert(str(sql_to_json_parser.extract_user_id(html)) == "15549606")   # check user id

    # check info box
    dict_obtained = sql_to_json_parser.extract_info_box(html)
    dict_expected = {'Distance' : '2.35 mi', 'Duration' : '24m:03s', 'Avg. Speed' : '10:13 min/mi', 'Max. Speed' : '7:01 min/mi', 'Calories' : '326 kcal', 'Hydration' : '0.34L', 'Min. Altitude' : '456 ft', 'Max. Altitude' : '577 ft', 'Total Ascent' : '417 ft', 'Total Descent' : '486 ft', 'Weather' : 'Sunny'}
    assert(dict_obtained == dict_expected)

    # check trace data
    dict_obtained = sql_to_json_parser.extract_trace_data(html)
    #f = open(os.path.join(cwd, "./data","1_trace.json"))
    #dict_expected = ujson.load(f, precise_float=True)
    #dict_expected = json.load(f)
    #dict_expected = {"data":[{"values":{"distance":0.0,"duration":0},"lng":17.04368,"lat":51.152705},{"values":{"distance":0.0,"duration":87219,"alt":556.10236},"lng":17.04368,"lat":51.152705},{"values":{"distance":0.036149167,"duration":95148,"alt":555.55554},"lng":17.043551458623856,"lat":51.15243885600381},{"values":{"pace":8.227328,"distance":0.053244255,"duration":103077,"alt":532.58966},"lng":17.043499743854625,"lat":51.1521968435004},{"values":{"pace":8.070909,"distance":0.07069855,"duration":111006,"alt":519.4663},"lng":17.043554545024904,"lat":51.1519473170012},{"values":{"pace":7.5802007,"distance":0.08712549,"duration":118935,"alt":508.53018},"lng":17.043595668761427,"lat":51.1517117193751},{"values":{"pace":7.370488,"distance":0.102969736,"duration":126864,"alt":515.09186},"lng":17.043631785640258,"lat":51.15148437120282}]}
    dict_expected = {"lat":[51.152705, 51.152705, 51.152439, 51.152197, 51.151947, 51.151712,51.151484], 
                    "lng":[17.04368, 17.04368, 17.043551, 17.043500, 17.043555, 17.043596, 17.043632],
                    "distance":[0.0, 0.0, 0.036149, 0.053244, 0.070699, 0.087125, 0.102970], 
                    "duration":[0, 87219, 95148, 103077, 111006, 118935, 126864],
                    "alt":['N', 556.10236, 555.55554, 532.58966, 519.4663, 508.53018, 515.09186],
                    "pace":['N', 'N', 'N', 8.227328, 8.070909, 7.580201, 7.370488]}
    f.close()
    assert(dict_obtained == dict_expected)

    # check date-time
    assert(sql_to_json_parser.extract_date_time(html) == "Apr 21, 2014 10:13 AM")

    # check sport type
    assert(sql_to_json_parser.extract_sport_type(html) == "Running")


def test_create_workout_dict():
    create_tmp_folder()
    cwd = os.path.dirname(os.path.abspath(__file__))
    user_id = "15549606"
    workout_id = "12345"
    sport = "Running"
    trace_dict = {
                    "lat":[51.152705, 51.152705, 51.152439, 51.152197, 51.151947, 51.151712,51.151484], 
                    "lng":[17.04368, 17.04368, 17.043551, 17.043500, 17.043555, 17.043596, 17.043632],
                    "distance":[0.0, 0.0, 0.036149, 0.053244, 0.070699, 0.087125, 0.102970], 
                    "duration":[0, 87219, 95148, 103077, 111006, 118935, 126864],
                    "alt":['N', 556.10236, 555.55554, 532.58966, 519.4663, 508.53018, 515.09186],
                    "pace":['N', 'N', 'N', 8.227328, 8.070909, 7.580201, 7.370488]
                }
    info_dict = { "Distance" : "2.35 mi", "Duration" : "24m:03s", "Avg. Speed" : "10:13 min/mi", "Max. Speed" : "7:01 min/mi", "Calories" : "326 kcal", "Hydration" : "0.34L", "Min. Altitude" : "456 ft", "Max. Altitude" : "577 ft", "Total Ascent" : "417 ft", "Total Descent" : "486 ft", "Weather" : "Sunny","date-time" : "Apr 21, 2014 10:13 AM", "user_id" : "15549606"}
    [w_dict, w_str, w_md5, status] = sql_to_json_parser.create_workout_dict(user_id, workout_id, sport, trace_dict, info_dict)
    assert (utils.get_workouts(os.path.join(cwd, "./data/2_expected_1.txt"))[0] == w_dict)


def small_workout_data_1():
    user_id = "15549606"
    workout_id = "12345"
    sport = "Running"
    trace_dict = {
                    "lat":[51.152705, 51.152705, 51.152439, 51.152197, 51.151947, 51.151712,51.151484], 
                    "lng":[17.04368, 17.04368, 17.043551, 17.043500, 17.043555, 17.043596, 17.043632],
                    "distance":[0.0, 0.0, 0.036149, 0.053244, 0.070699, 0.087125, 0.102970], 
                    "duration":[0, 87219, 95148, 103077, 111006, 118935, 126864],
                    "alt":['N', 556.10236, 555.55554, 532.58966, 519.4663, 508.53018, 515.09186],
                    "pace":['N', 'N', 'N', 8.227328, 8.070909, 7.580201, 7.370488]
                }
    info_dict = { "Distance" : "2.35 mi", "Duration" : "24m:03s", "Avg. Speed" : "10:13 min/mi", "Max. Speed" : "7:01 min/mi", "Calories" : "326 kcal", "Hydration" : "0.34L", "Min. Altitude" : "456 ft", "Max. Altitude" : "577 ft", "Total Ascent" : "417 ft", "Total Descent" : "486 ft", "Weather" : "Sunny","date-time" : "Apr 21, 2014 10:13 AM", "user_id" : "15549606"}
    return [user_id, workout_id, sport, trace_dict, info_dict]

def create_tmp_folder():
    p = "/tmp/fitness"
    if (not os.path.isdir(p)):
        os.mkdir(p)

def test_write_workout():
    cwd = os.path.dirname(os.path.abspath(__file__))
    create_tmp_folder()
    
    # first test one write
    [user_id, workout_id, sport, trace_dict, info_dict] = small_workout_data_1()
    fd = open("/tmp/fitness/1.txt","w")
    [w_dict, w_str, w_md5, status] = sql_to_json_parser.create_workout_dict(user_id, workout_id, sport, trace_dict, info_dict)
    assert(sql_to_json_parser.write_workout(w_dict, w_str, w_md5, fd))   # should return True
    fd.close()
    assert(utils.get_workouts("/tmp/fitness/1.txt") == utils.get_workouts(os.path.join(cwd,"./data/2_expected_1.txt")))

    # now test 2nd write, which should get detected as a duplicate
    #[user_id, workout_id, sport, trace_dict, info_dict] = small_workout_data_1()
    #[w_dict, w_str, w_md5] = sql_to_json_parser.create_workout_dict(user_id, workout_id, sport, trace_dict, info_dict)
    #fd = open("/tmp/fitness/1.txt","a")
    #assert(sql_to_json_parser.write_workout(w_dict, w_str, w_md5, fd) == False)
    #fd.close()

    # now try writing a different workout
    [user_id, workout_id, sport, trace_dict, info_dict] = small_workout_data_1()
    info_dict["Weather"] = "Raining"
    [w_dict, w_str, w_md5, status] = sql_to_json_parser.create_workout_dict(user_id, workout_id, sport, trace_dict, info_dict)
    fd = open("/tmp/fitness/1.txt","a")
    assert(sql_to_json_parser.write_workout(w_dict, w_str, w_md5, fd))   # should return True
    fd.close()
    assert(utils.get_workouts("/tmp/fitness/1.txt") == utils.get_workouts(os.path.join(cwd, "./data/2_expected_2.txt")))

"""
def test_add_workout_to_user():
    outfolder = "/tmp/fitness"
    if (os.path.isdir(outfolder)):
        shutil.rmtree(outfolder)
    os.mkdir(outfolder)
    cwd = os.path.dirname(os.path.abspath(__file__))
    #dicts = utils.get_workouts(os.path.join(cwd, "./data/2_expected_1.txt"))
    #info_dict = dicts[0]
    #trace_dict = {}
    #trace_dict['data'] = info_dict['data']
    #del info_dict['data']
    user_id = "15549606"
    workout_id = "12345"
    sport = "Running"
    trace_dict = {
                    "lat":[51.152705, 51.152705, 51.152439, 51.152197, 51.151947, 51.151712,51.151484], 
                    "lng":[17.04368, 17.04368, 17.043551, 17.043500, 17.043555, 17.043596, 17.043632],
                    "distance":[0.0, 0.0, 0.036149, 0.053244, 0.070699, 0.087125, 0.102970], 
                    "duration":[0, 87219, 95148, 103077, 111006, 118935, 126864],
                    "alt":['N', 556.10236, 555.55554, 532.58966, 519.4663, 508.53018, 515.09186],
                    "pace":['N', 'N', 'N', 8.227328, 8.070909, 7.580201, 7.370488]
                }
    info_dict = { "Distance" : "2.35 mi", "Duration" : "24m:03s", "Avg. Speed" : "10:13 min/mi", "Max. Speed" : "7:01 min/mi", "Calories" : "326 kcal", "Hydration" : "0.34L", "Min. Altitude" : "456 ft", "Max. Altitude" : "577 ft", "Total Ascent" : "417 ft", "Total Descent" : "486 ft", "Weather" : "Sunny","date-time" : "Apr 21, 2014 10:13 AM"}
    
    # test writing first workout
    p = SqlToJsonParser()
    assert(p.add_workout_to_user(user_id, workout_id, sport, trace_dict, info_dict, outfolder))
    assert(os.path.isfile("/tmp/fitness/155/15549606.txt"))
    #print utils.get_workouts("/tmp/fitness/155/15549606.txt")[0]
    #print utils.get_workouts(os.path.join(cwd, "./data/2_expected_1.txt"))[0]
    for k in utils.get_workouts("/tmp/fitness/155/15549606.txt")[0]:
        if (k not in utils.get_workouts(os.path.join(cwd, "./data/2_expected_1.txt"))[0]):
            print k
    assert(utils.get_workouts("/tmp/fitness/155/15549606.txt")[0]["lng"] == utils.get_workouts(os.path.join(cwd, "./data/2_expected_1.txt"))[0]["lng"])
    assert(utils.get_workouts("/tmp/fitness/155/15549606.txt")[0]["pace"] == utils.get_workouts(os.path.join(cwd, "./data/2_expected_1.txt"))[0]["pace"])
    assert(utils.get_workouts("/tmp/fitness/155/15549606.txt")[0]["duration"] == utils.get_workouts(os.path.join(cwd, "./data/2_expected_1.txt"))[0]["duration"])
    assert(utils.get_workouts("/tmp/fitness/155/15549606.txt")[0]["distance"] == utils.get_workouts(os.path.join(cwd, "./data/2_expected_1.txt"))[0]["distance"])
    assert(utils.get_workouts("/tmp/fitness/155/15549606.txt")[0]["alt"] == utils.get_workouts(os.path.join(cwd, "./data/2_expected_1.txt"))[0]["alt"])
    assert(utils.get_workouts("/tmp/fitness/155/15549606.txt")[0]["lat"] == utils.get_workouts(os.path.join(cwd, "./data/2_expected_1.txt"))[0]["lat"])
    assert(utils.get_workouts("/tmp/fitness/155/15549606.txt")[0] == utils.get_workouts(os.path.join(cwd, "./data/2_expected_1.txt"))[0])

    # test writing 2nd workout (duplicate)
    # its better to explicitly initialize everything again since some of the dictionaries get modified internally in the add_workout_to_user function
    user_id = "15549606"
    workout_id = "12345"
    sport = "Running"
    trace_dict = {
                    "lat":[51.152705, 51.152705, 51.152439, 51.152197, 51.151947, 51.151712,51.151484], 
                    "lng":[17.04368, 17.04368, 17.043551, 17.043500, 17.043555, 17.043596, 17.043632],
                    "distance":[0.0, 0.0, 0.036149, 0.053244, 0.070699, 0.087125, 0.102970], 
                    "duration":[0, 87219, 95148, 103077, 111006, 118935, 126864],
                    "alt":['N', 556.10236, 555.55554, 532.58966, 519.4663, 508.53018, 515.09186],
                    "pace":['N', 'N', 'N', 8.227328, 8.070909, 7.580201, 7.370488]
                }
    info_dict = { "Distance" : "2.35 mi", "Duration" : "24m:03s", "Avg. Speed" : "10:13 min/mi", "Max. Speed" : "7:01 min/mi", "Calories" : "326 kcal", "Hydration" : "0.34L", "Min. Altitude" : "456 ft", "Max. Altitude" : "577 ft", "Total Ascent" : "417 ft", "Total Descent" : "486 ft", "Weather" : "Sunny","date-time" : "Apr 21, 2014 10:13 AM"}
    assert (p.add_workout_to_user(user_id, workout_id, sport, trace_dict, info_dict, outfolder) == False)   # False coz we are adding duplicate
    
    # now write a 2nd different workout
    info_dict['Weather'] = "Raining"    # to generate a different workout
    assert (p.add_workout_to_user(user_id, workout_id, sport, trace_dict, info_dict, outfolder))
    assert(os.path.isfile("/tmp/fitness/155/15549606.txt"))
    assert(utils.get_workouts("/tmp/fitness/155/15549606.txt") == utils.get_workouts(os.path.join(cwd, "./data/2_expected_2.txt")))
"""

def test_parse_html():
    #outfolder = "/tmp/fitness"
    #if (os.path.isdir(outfolder)):
        #shutil.rmtree(outfolder)
    #os.mkdir(outfolder)
    cwd = os.path.dirname(os.path.abspath(__file__))
    f = open(os.path.join(cwd, "data","1.html"))
    html = f.read()
    f.close()
    workout_id = "12345"
    [w_dict, w_str, w_md5, status] = sql_to_json_parser.parse_html(workout_id, html)
    #assert(os.path.isfile("/tmp/fitness/155/15549606.txt"))
    assert(w_dict == utils.get_workouts(os.path.join(cwd, "./data/2_expected_1.txt"))[0])

def test_reformat_trace_data():
    d = {"data": [{"lat": 50.223, "lng": 19.247, "values": {"duration": 0, "distance": 0.0}}, {"lat": 50.22, "lng": 19.24, "values": {"duration": 19352, "distance": 0.004, "alt": 1040.02, "speed": 0.830}}, {"lat": 50.22, "lng": 19.24, "values": {"duration": 29028, "distance": 0.0096, "alt": 1037.12, "speed": 1.5}}]}
    expected_d = {"lat":[50.223, 50.22, 50.22], "lng" : [19.247, 19.24, 19.24], "duration":[0, 19352, 29028], "distance":[0.0, 0.004, 0.0096,], "speed":['N', 0.830, 1.5], "alt":['N', 1040.02, 1037.12,]}
    observed_d = sql_to_json_parser.reformat_trace_data(d)
    assert(expected_d == observed_d)
    
    # add one test to check for unknown traces
    d = {"data": [{"lat": 50.223, "lng": 19.247, "values": {"duration": 0, "distance": 0.0}}, {"lat": 50.22, "lng": 19.24, "height":12.32 ,"values": {"duration": 19352, "distance": 0.004, "alt": 1040.02, "speed": 0.830}}, {"lat": 50.22, "lng": 19.24, "values": {"duration": 29028, "distance": 0.0096, "alt": 1037.12, "speed": 1.5}}]}
    try:
        sql_to_json_parser.reformat_trace_data(d)    # should throw an exception for  "height"
        assert(False)
    except:
        pass

def test_get_workouts():
    cwd = os.path.dirname(os.path.abspath(__file__))
    expected_dicts = []
    expected_dicts.append({"workout_id" : "12345", "user_id" : "15549606",
                    "lat":[51.152705, 51.152705, 51.152439, 51.152197, 51.151947, 51.151712,51.151484], 
                    "lng":[17.04368, 17.04368, 17.043551, 17.043500, 17.043555, 17.043596, 17.043632],
                    "distance":[0.0, 0.0, 0.036149, 0.053244, 0.070699, 0.087125, 0.102970], 
                    "duration":[0, 87219, 95148, 103077, 111006, 118935, 126864],
                    "alt":['N', 556.10236, 555.55554, 532.58966, 519.4663, 508.53018, 515.09186],
                    "pace":['N', 'N', 'N', 8.227328, 8.070909, 7.580201, 7.370488],
                    "Distance" : "2.35 mi", "Duration" : "24m:03s", "Avg. Speed" : "10:13 min/mi", "Max. Speed" : "7:01 min/mi", "Calories" : "326 kcal", "Hydration" : "0.34L", "Min. Altitude" : "456 ft", "Max. Altitude" : "577 ft", "Total Ascent" : "417 ft", "Total Descent" : "486 ft", "Weather" : "Sunny","date-time" : "Apr 21, 2014 10:13 AM", "workout_id" : "12345", "sport" : "Running"})
    expected_dicts.append({"workout_id" : "12345", "user_id" : "15549606",
                    "lat":[51.152705, 51.152705, 51.152439, 51.152197, 51.151947, 51.151712,51.151484], 
                    "lng":[17.04368, 17.04368, 17.043551, 17.043500, 17.043555, 17.043596, 17.043632],
                    "distance":[0.0, 0.0, 0.036149, 0.053244, 0.070699, 0.087125, 0.102970], 
                    "duration":[0, 87219, 95148, 103077, 111006, 118935, 126864],
                    "alt":['N', 556.10236, 555.55554, 532.58966, 519.4663, 508.53018, 515.09186],
                    "pace":['N', 'N', 'N', 8.227328, 8.070909, 7.580201, 7.370488],
                    "Distance" : "2.35 mi", "Duration" : "24m:03s", "Avg. Speed" : "10:13 min/mi", "Max. Speed" : "7:01 min/mi", "Calories" : "326 kcal", "Hydration" : "0.34L", "Min. Altitude" : "456 ft", "Max. Altitude" : "577 ft", "Total Ascent" : "417 ft", "Total Descent" : "486 ft", "Weather" : "Raining","date-time" : "Apr 21, 2014 10:13 AM", "workout_id" : "12345", "sport" : "Running"})
    assert(utils.get_workouts(os.path.join(cwd, "./data/2_expected_2.txt"))  == expected_dicts)

def test_combine_gzip_files():
    
    file1 = "/tmp/fitness/1.gz"
    file2 = "/tmp/fitness/2.gz"

    [user_id, workout_id, sport, trace_dict, info_dict] = small_workout_data_1()
    fd = gzip.open(file1,"w")
    [w_dict1, w_str, w_md5, status] = sql_to_json_parser.create_workout_dict(user_id, workout_id, sport, trace_dict, info_dict)
    assert(sql_to_json_parser.write_workout(w_dict1, w_str, w_md5, fd))   # should return True
    fd.close()
    assert(os.path.isfile(file1))

    [user_id, workout_id, sport, trace_dict, info_dict] = small_workout_data_1()
    fd = gzip.open(file2,"w")
    sport = "Running"
    info_dict["Weather"] = "Raining"
    [w_dict2, w_str, w_md5, status] = sql_to_json_parser.create_workout_dict(user_id, workout_id, sport, trace_dict, info_dict)
    assert(sql_to_json_parser.write_workout(w_dict2, w_str, w_md5, fd))   # should return True
    fd.close()
    assert(os.path.isfile(file2))

    files = [file1, file2]
    outfile = "/tmp/fitness/3.gz"
    utils.combine_gzip_files(files, outfile)
    assert(os.path.isfile(outfile))
    
    w_dicts = utils.get_workouts(outfile)
    assert(len(w_dicts) == 2)
    assert(w_dict1 in w_dicts)
    assert(w_dict2 in w_dicts)

def test_parser():
    create_tmp_folder()
    cwd = os.path.dirname(os.path.abspath(__file__))
    infile = os.path.join(cwd, "data/endoMondoSmall.sql.gz")
    outfile = "/tmp/fitness/endoMondoSmall.gz"
    p = SqlToJsonParser(infile=infile,outfile=outfile, nprocesses=2)
    p.run()
    s = p.get_stats()
    assert(s.workouts == 17)
    assert(s.workouts_invalid == 1)
    assert(s.lines_parsed == 2)
    assert(s.workouts_without_data == 0)
    assert(s.workouts_without_info == 0)

if __name__ == "__main__":
    test_extract()
    test_create_workout_dict()
    test_write_workout()
    #test_add_workout_to_user()
    test_parse_html()
    test_reformat_trace_data()
    test_get_workouts()
    test_combine_gzip_files()
    test_parser()
