import gzip
import utils
import sys

wids = set()
with gzip.open(sys.argv[1]) as f:
    n = 0
    for line in f:
        d = utils.json_to_dict(line)
        w = d["workout_id"]
        if (w in wids):
            print "DUPLICATE FOUND.. workout id = " + str(w)
        wids.add(w)
        n += 1
        if (n % 100000 == 0):
            print "Done with %d workouts.." % (n)
