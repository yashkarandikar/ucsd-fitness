#!/usr/bin/python

import argparse
import gzip
import numpy as np
import utils
import matplotlib.pyplot as plt
import os 

def sort_avg_data(x, y):
    """
    Sort lists in x and y considering x as the keys
    x is a list of lists. Each list is for one parameter, and contains average values of that parameter over all workouts
    y is a list of lists. Each list is for one parameter, and contains average values of that parameter over all workouts
    """
    n_params = len(x)
    for i in range(0, n_params):
        xy = zip(x[i], y[i])
        xy.sort()
        x[i] = [a for a, b in xy]
        y[i] = [b for a, b in xy]
    return [x, y]


def get_avg_data(infile, x_params, y_params):
    """
    returns data averaged over each workout. For each i, returns y_params[i] and x_params[i] averaged over each workout. Thus each point in the plot corresponds to one workout
    x_params  list of parameters, must be present in the data
    y_params  list of parameters, must be present in the data
    Must be true : len(x_params) == len(y_params)
    """
    assert(len(x_params) == len(y_params))
    assert(len(x_params) > 0)
    assert(len(y_params) > 0)

    n_params = len(x_params)
    x = []; y = []   # will be a list of lists, one list for each param. Each list is the avg. values of that param over each workout
    for i in range(0, n_params):
        x.append([])
        y.append([])

    print "X parameters : " + str(x_params)
    print "Y parameters : " + str(y_params)
    n_lines = 0

    infile_basename, ext = os.path.splitext(infile)
    if (ext == ".gz"):
        f = gzip.open(infile)
    elif(ext == ".txt"):
        f = open(infile)
    else:
        raise Exception("File format not recognized")

    for line in f:
        # each line is a workout
        w = utils.json_to_dict(line.strip())
        for i in range(0, n_params):
            # for each parameter
            #print i
            xp = x_params[i]; yp = y_params[i]      # x and y axis parameters
            if (w.has_key(xp) and w.has_key(yp)):
                #print "workout " + str(w["workout_id"]) + " has both.. "  + xp + " and " + yp
                [x_trace, y_trace] = utils.remove_null_values(w[xp], w[yp])
                mx = np.mean(x_trace)
                my = np.mean(y_trace)
                x[i].append(mx)    # append the mean value of that param over the workout
                y[i].append(my)
        n_lines += 1
        #if (n_lines == 1000):
        #    break

    f.close()
    
    # print stats
    print "{:<25}{:<15}".format("Plot", "# Workouts")
    for i in range(0, n_params):
        print "{:<25}{:<15}".format(y_params[i] + " vs " + x_params[i], len(x[i]))

    return [x, y]


def plot_avg_data(x, y, x_params, y_params):
    assert(len(x_params) == len(y_params))
    for i in range(len(x_params)):
        plt.figure(i)
        plt.plot(x[i], y[i])
        plt.xlabel(x_params[i])
        plt.ylabel(y_params[i])
    plt.show()

def visualize_all(infile):
    x_params = ["alt", "hr", "duration", "distance"]
    #y_params = ["pace", "pace", "pace", "pace"]
    y_params = ["speed", "speed", "speed", "speed"]
    [x, y] = get_avg_data(infile, x_params, y_params)
    [x, y] = sort_avg_data(x, y)
    plot_avg_data(x, y, x_params, y_params)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='See various plots over all workouts')
    parser.add_argument('--infile', type=str, help='.gz file or .txt containing all workouts', dest='infile')
    args = parser.parse_args()
    if (args.infile is None):
        parser.print_usage()
        exit(0)
    visualize_all(args.infile)

