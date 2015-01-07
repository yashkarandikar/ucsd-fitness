import os
import sys
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath,'..'))
from condense_and_clean_data import condense_and_clean_data
import utils

def create_tmp_folder():
    p = "/tmp/fitness"
    if (not os.path.isdir(p)):
        os.mkdir(p)

def test_condense_and_clean_data():
    create_tmp_folder()
    cwd = os.path.dirname(os.path.abspath(__file__))
    infile = os.path.join(cwd, "./data","workouts.gz")
    outfile = "/tmp/fitness/condense.gz"

    expected_workouts = [{"Distance": 2.35, "user_id": "15549606", "Max. Speed": 7.016667, "lat(avg)":51.1522, "workout_id": "327000000", "distance(avg)": 0.070037, "Avg. Speed": 10.216667, "Calories": 326, "Min. Altitude": 456,"Weather": "Sunny", "sport": "Running", "duration(avg)": 103077, "Hydration": 0.34, "Total Ascent": 417, "Total Descent": 486, "Duration": 1443, "pace(avg)": 7.812231, "lng(avg)": 17.044890, "date-time": "Apr 21, 2014 10:13 AM", "Max. Altitude": 577},
                        {"Distance": 3.35, "user_id": "15549607", "Max. Speed": 5.016667, "lat(avg)": 5.1522, "workout_id": "327000001", "distance(avg)": 0.076608, "Avg. Speed": 11.216667, "Calories": 400, "Min. Altitude": 856, "Weather": "Raining", "sport": "Running", "duration(avg)":103127.8, "Hydration": 0.56, "Total Ascent": 500, "Total Descent": 600, "Duration": 1803, "pace(avg)": 8.15471, "lng(avg)": 17.047410, "date-time": "Oct 21, 2014 10:10 PM", "Max. Altitude": 650},
                        {"Distance": 4.35, "user_id": "15549608", "Max. Speed": 9.016667, "lat(avg)": 31.1522, "workout_id": "327000002", "distance(avg)": 0.076088, "Avg. Speed": 12.216667, "Calories": 500, "Min. Altitude": 500, "Weather": "Sunny", "sport": "Running", "duration(avg)": 105237.2, "Hydration": 0.53, "Total Ascent": 417, "Total Descent": 421, "Duration": 1672, "pace(avg)": 7.892475, "lng(avg)": 17.200890, "date-time": "Jun 1, 2014 9:13 AM", "Max. Altitude": 900}]
    condense_and_clean_data(infile, outfile)
    obtained_workouts = utils.get_workouts(outfile)
    assert(expected_workouts == obtained_workouts)
    return True

if __name__ == "__main__":
    test_condense_and_clean_data()
