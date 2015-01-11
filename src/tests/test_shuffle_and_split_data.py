import os
import sys
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath,'..'))
import utils
from shuffle_and_split_data import shuffle_and_split_data, split_data

def create_tmp_folder():
    p = "/tmp/fitness"
    if (not os.path.isdir(p)):
        os.mkdir(p)

def test_shuffle_and_split_data():
    cwd = os.path.dirname(os.path.abspath(__file__))
    infile = os.path.join(cwd, "./data","workouts.gz")
    create_tmp_folder()
    outfile_base = "/tmp/workouts_"
    shuffle_and_split_data(infile, outfile_base, infile_line_count = 3)
    outfile1 = outfile_base + "1.gz"
    outfile2 = outfile_base + "2.gz"

    ws_expected = utils.get_workouts(infile)
    ws_part1 = utils.get_workouts(outfile1)
    ws_part2 = utils.get_workouts(outfile2)
    ws_obtained = ws_part1 + ws_part2

    assert(len(ws_expected) == len(ws_obtained))
    for w in ws_expected:
        assert(w in ws_obtained)
    for w in ws_obtained:
        assert(w in ws_expected)

def test_split_data():
    cwd = os.path.dirname(os.path.abspath(__file__))
    infile = os.path.join(cwd, "./data","workouts.gz")
    create_tmp_folder()
    outfile_base = "/tmp/workouts_"
    split_data(infile, outfile_base, infile_line_count = 3)
    outfile1 = outfile_base + "1.gz"
    outfile2 = outfile_base + "2.gz"

    ws_expected = utils.get_workouts(infile)
    ws_part1 = utils.get_workouts(outfile1)
    ws_part2 = utils.get_workouts(outfile2)
    ws_obtained = ws_part1 + ws_part2
    assert(ws_expected == ws_obtained)
    assert(ws_expected[0] == ws_obtained[0])
    assert(ws_expected[1] == ws_obtained[1])
    assert(ws_expected[2] == ws_obtained[2])

if __name__ == "__main__":
    test_shuffle_and_split_data()
    test_split_data()
