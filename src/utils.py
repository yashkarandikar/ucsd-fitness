import simplejson as json
import os
import gzip

def get_workouts(infile):
    # assumes each line of the input file is a json structure
    dicts = []
    fName, fExt = os.path.splitext(infile) 
    f = 0
    if (fExt == ".gz"):
        f = gzip.open(infile)
    elif (fExt == ".txt"):
        f = open(infile)
    else:
        print fExt
        raise Exception("Invalid file format")
    
    for line in f:
        dicts.append(json.loads(line))
    f.close()
    return dicts

def json_to_dict(s):
    # s is a json formatted string
    return json.loads(s)

def remove_null_values(l):
    # given list of values, removes those marked 'N'
    return [x for x in l if x != 'N']

def remove_null_values(l1, l2):
    # given list of values, removes those marked 'N'
    assert(len(l1) == len(l2))
    n = len(l1)
    l1_new = []
    l2_new = []
    for i in range(0, n):
        if (l1[i] != 'N' and l2[i] != 'N'):
            l1_new.append(l1[i])
            l2_new.append(l2[i])
    return [l1_new, l2_new]

def get_user_id_from_filename(infile):
    f = os.path.basename(infile)
    parts = f.split(".")
    if (len(parts) != 2):
        raise Exception("Filename is not in recognized format")
    return int(parts[0])

def combine_gzip_files(files, outfile):
    # combines multiple gzip files into one single gzipped file
    command = "cat"
    for f in files:
        command = command + " " + f
    command = command + " > " + outfile
    os.system(command)
