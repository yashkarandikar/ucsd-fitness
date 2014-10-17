import ujson
import os

def json_to_dicts(infile):
    # assumes each line of the input file is a json structure
    dicts = []
    f = open(infile)
    with open(infile) as f:
        for line in f:
            dicts.append(ujson.loads(line, precise_float=True))
    return dicts


def get_user_id_from_filename(infile):
    f = os.path.basename(infile)
    parts = f.split(".")
    if (len(parts) != 2):
        raise Exception("Filename is not in recognized format")
    return int(parts[0])
