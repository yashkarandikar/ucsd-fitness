#!/usr/bin/python

import sys
import os.path
import simplejson as json
import utils 
import argparse
import gzip
import time

def extract_part_of_file(infile, start_line = None, end_line = None, outfile = None):
    start_time = time.time()
    if (infile == ""):
        raise Exception("Input file not supplied")
    if (not os.path.isfile(infile)):
        print "File not found.."
        exit(0)

    if (start_line == None):
        start_line = 0
    if (end_line == None):
        end_line = float("inf")
    if (outfile == None):
        s = "_" + str(start_line) + "_" + str(end_line)
        outfile = utils.append_to_base_filename(infile, s)

    print "Input file : " + infile
    print "Output file : " + outfile
    print "Start line number : " + str(start_line)
    print "End line number : " + str(end_line)

    outf = gzip.open(outfile, "w")

    print "Reading file " + infile
    # now read input file
    with gzip.open(infile) as f:
        i_line = 0
        for line in f:
            i_line += 1
            if (i_line > end_line):
                break
            if (i_line >= start_line and i_line <= end_line):
                outf.write(line)
            
            if (i_line % 10000 == 0):
                print "Done reading line number ", i_line
    
    outf.close()

    print "Done.."
    end_time = time.time()
    print "Total time taken = ", end_time - start_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extracts part of .gz file')
    parser.add_argument('--infile', type=str, help='.gz file', dest='infile')
    parser.add_argument('--outfile', type=str, help='.gz file', dest='outfile')
    parser.add_argument('--start', type=int, help='start line number', dest='start_line')
    parser.add_argument('--end', type=int, help='end line number', dest='end_line')
    args = parser.parse_args()

    #check if all required options are available
    if (args.infile is None):
        parser.print_usage()
        sys.exit(0)

    extract_part_of_file(infile = args.infile, outfile = args.outfile, start_line = args.start_line, end_line = args.end_line)
