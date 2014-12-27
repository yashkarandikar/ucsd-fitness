import os
import gzip
import utils
import numpy
import argparse

def condense(infile, outfile):
    """
    infile must be a .gz file generated by the sql_to_json_parser.py
    condense will replace trace data by averages
    """
    fo = gzip.open(outfile, "w")
    fi = gzip.open(infile)
    n = 0
    for line in fi:
        d = {}
        w = utils.json_to_dict(line.strip())
        for k, v in w.items():
            if (isinstance(v, list)):
                v = numpy.mean(utils.remove_null_values_single(v))
            d[k] = v
        w_str = utils.dict_to_json(d)
        fo.write(w_str + "\n")
        n += 1
        if (n % 1000 == 0):
            print "Written %d workouts.." % (n)
    fi.close()
    fo.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reads .gz file written by sql_to_json_parser.py and condenses all traces to average values')
    parser.add_argument('--infile', type=str, help='.gz file', dest='infile')
    parser.add_argument('--outfile', type=str, help='.gz file', dest='outfile')
    args = parser.parse_args()
    if (args.infile is None or args.outfile is None):
        parser.print_usage()
    else:
        condense(args.infile,args.outfile)

