import gzip 
import argparse
import numpy as np
import math
import time

def count_lines_in_file(infile):
    n = 0
    with gzip.open(infile) as f:
        for line in f:
            n += 1
    return n

def split_data(infile, outfile_base, fraction = None, infile_line_count = None):
    print "Shuffling disabled.."

    # get number of lines in file
    N = 0
    if (infile_line_count is not None):
        N = int(infile_line_count)
    else:
        # count number of lines in file
        print "Counting lines in file.."
        N = count_lines_in_file(infile)
    print "File has %d lines (workouts).." % (N)

    # split into 2 parts
    if (fraction == None):
        fraction = 0.5
    assert(0.0 <= fraction and fraction <= 1.0)
    end1 = math.ceil(float(N) * fraction)

    # writing
    outf1 = gzip.open(outfile_base + "1.gz", "w")
    outf2 = gzip.open(outfile_base + "2.gz", "w")
    print "Writing 2 sets to files.."
    with gzip.open(infile) as f:
        n_line = 0
        n1 = n2 = 0
        while n_line < end1:
            outf1.write(f.readline())
            n1 += 1
            n_line += 1
            if (n_line % 100000 == 0):
                print "Done writing total %d workouts.." % (n_line)
        while n_line < N:
            outf2.write(f.readline())
            n2 += 1
            n_line += 1
            if (n_line % 100000 == 0):
                print "Done writing total %d workouts.." % (n_line)
    
    outf1.close()
    outf2.close()

    print "Written %d workouts to set 1 and %d workouts to set 2.." % (n1, n2)
    assert(n1 + n2 == N)

def shuffle_and_split_data(infile, outfile_base, fraction = None, infile_line_count = None):
    print "Shuffling enabled.."

    # get number of lines in file
    N = 0
    if (infile_line_count is not None):
        N = int(infile_line_count)
    else:
        # count number of lines in file
        print "Counting lines in file.."
        N = count_lines_in_file(infile)
    print "File has %d lines (workouts).." % (N)

    # generate a permutation if required
    print "Generating permutation.."
    perm = np.random.permutation(N)

    # split the permutation into 2 parts, corresponding to the 2 sets
    if (fraction == None):
        fraction = 0.5
    assert(0.0 <= fraction and fraction <= 1.0)
    end1 = math.ceil(float(N) * fraction)
    perm1 = perm[0:end1]
    perm2 = perm[end1:]
    np1 = len(perm1) 
    np2 = len(perm2)
    assert(np1 + np2 == N)
    print "Set 1 will have %d workouts, set 2 will have %d workouts" % (len(perm1), len(perm2))
    
    # generate 2 partitions of the data from the 2 permutations
    belongs_to = [0] * N
    for x in perm2:
        belongs_to[x] = 1
    del perm1
    del perm2
    outf1 = gzip.open(outfile_base + "1.gz", "w")
    outf2 = gzip.open(outfile_base + "2.gz", "w")
    outf = [outf1, outf2] 
    print "Writing 2 sets to files.."
    with gzip.open(infile) as f:
        n_line = 0
        n = [0, 0]
        for line in f:
            outf[belongs_to[n_line]].write(line)
            n[belongs_to[n_line]] += 1
            n_line += 1
            if (n_line % 100000 == 0):
                print "Done writing total %d workouts.." % (n_line)
    
    outf1.close()
    outf2.close()

    print "Written %d workouts to set 1 and %d workouts to set 2.." % (n[0], n[1])

    assert(np1 == n[0] and np2 == n[1])
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reads .gz file written by sql_to_json_parser.py, (optionally) shuffles and splits into 2 sets')
    parser.add_argument('--infile', type=str, help='.gz file', dest='infile')
    parser.add_argument('--infile-line-count', type=int, help='number of lines (workouts) in infile.. This is an optimization.', dest='infile_line_count')
    parser.add_argument('--split-fraction', type=float, help='split fraction (must be between 0 and 1)', dest='split_fraction')
    parser.add_argument('--outfile-base', type=str, help='the 2 output files will be named <outfile_base>1.gz and <outfile_base.npy>2.gz', dest='outfile_base')
    parser.add_argument('--no-shuffle', action='store_true', help='disable shuffling (default: False i.e. shuffling is ON by default)', default=False, dest='no_shuffle')
    args = parser.parse_args()
    if (args.infile is None or args.outfile_base is None):
        parser.print_usage()
    else:
        t1 = time.time()
        shuffle = not args.no_shuffle
        if (shuffle):
            shuffle_and_split_data(infile = args.infile, outfile_base = args.outfile_base, fraction = args.split_fraction, infile_line_count = args.infile_line_count)
        else:
            split_data(infile = args.infile, outfile_base = args.outfile_base, fraction = args.split_fraction, infile_line_count = args.infile_line_count)
        t2 = time.time()
        print "Time taken = %d" % (t2 - t1)

