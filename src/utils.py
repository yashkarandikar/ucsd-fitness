import simplejson as json
import os

def json_lines_to_dicts(infile):
    # assumes each line of the input file is a json structure
    dicts = []
    f = open(infile)
    with open(infile) as f:
        for line in f:
            #dicts.append(ujson.loads(line, precise_float=True))
            dicts.append(json.loads(line))
    return dicts

def json_to_dict(s):
    # s is a json formatted string
    return json.loads(s)

def dict_to_json(d):
    # d is a dictionary
    return json.dumps(d)

def remove_null_values_single(l):
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

class Unit(object):
    params = {"alt":"ft",
                "distance" : "mi",
                "hr" : "bpm",
                "speed" : "mph",
                "pace" : "min/mi"}
    @staticmethod
    def get(param):
        if (Unit.params.has_key(param)):
            return Unit.params[param]
        else:
            return "UNK"
