import argparse
import gzip
import utils
import operator

def print_stats(d, count, col1, col2):
    print ""
    d = sorted(d.items(), key=operator.itemgetter(1))
    d.reverse()
    print "{:<20}{:<20}".format(col1, col2)
    print "-"*40
    lim = count
    for (k,v) in d:
        print "{:<20}{:<20}".format(k, v)
        lim -= 1
        if (lim == 0):
            break


def get_stats(infile):
    workouts_for_param = {}
    workouts_for_sport = {}
    workouts_for_user = {}
    with gzip.open(infile) as f:
        nlines = 0
        for line in f:
            d = utils.json_to_dict(line)

            # workouts per param
            for k in d.keys():
                if (not workouts_for_param.has_key(k)):
                    workouts_for_param[k] = 0
                workouts_for_param[k] += 1

            # workouts per sport type
            if (d.has_key("sport")):
                sport = d["sport"]
                if (not workouts_for_sport.has_key(sport)):
                    workouts_for_sport[sport] = 0
                workouts_for_sport[sport] += 1

            # workouts per user
            user = d["user_id"]
            if (not workouts_for_user.has_key(user)):
                workouts_for_user[user] = 0
            workouts_for_user[user] += 1

            nlines += 1
            if (nlines % 100000 == 0):
                print "Done with %d workouts.." % (nlines)

    # print stats
    print_stats(workouts_for_param, 100, "Parameter", "# Workouts")
    print_stats(workouts_for_user, 100, "User ID", "# Workouts")
    print_stats(workouts_for_sport, 100, "Sport", "# Workouts")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='See various plots over all workouts')
    parser.add_argument('--infile', type=str, help='.gz file or .txt containing all workouts', dest='infile')
    args = parser.parse_args()
    if (args.infile is None):
        parser.print_usage()
        exit(0)
    get_stats(args.infile)
