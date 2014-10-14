import ujson

def json_to_dicts(infile):
    # assumes each line of the input file is a json structure
    dicts = []
    f = open(infile)
    with open(infile) as f:
        for line in f:
            dicts.append(ujson.loads(line, precise_float=True))
    return dicts
