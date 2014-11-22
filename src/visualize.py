#!/usr/bin/python

import argparse
import gzip
import numpy as np
import utils
import matplotlib.pyplot as plt
import os 

class Data(object):

    param_ranges = {"distance" : [0, 1000]}   # static dictionary

    def __init__(self, sport, xparam, yparam):
        self.xparam = xparam
        self.yparam = yparam
        self.sport = sport
        self.xvals = []
        self.yvals = []

    def add_point(self, x, y):
        self.xvals.append(x)
        self.yvals.append(y)

    def sort(self):
        xy = zip(self.xvals, self.yvals)
        xy.sort()
        self.xvals = [a for a, b in xy]
        self.yvals = [b for a, b in xy]

    def plot(self):
        assert(len(self.xvals) == len(self.yvals))
        self.sort()
        plt.figure()
        plt.plot(self.xvals, self.yvals)
        xlims = Data.param_range(self.xparam)
        ylims = Data.param_range(self.yparam)
        if (xlims is not None):
            plt.xlim(xlims)
        if (ylims is not None):
            plt.ylim(ylims)
        plt.xlabel(self.xparam + " (%s)"%(utils.Unit.get(self.xparam)))
        plt.ylabel(self.yparam + " (%s)"%(utils.Unit.get(self.yparam)))
        plt.title("Sport: %s"%(self.sport))

    def empty(self):
        return (len(self.xvals) == 0)

    def summary(self):
        return "{:<15}{:<15}{:<15}{:<15}".format(self.sport, self.xparam, self.yparam, len(self.xvals))

    @staticmethod
    def summary_format():
        return "{:<15}{:<15}{:<15}{:<15}".format("Sport","X param","Y param","# workouts")

    @staticmethod
    def param_range(param):
        if (Data.param_ranges.has_key(param)):
            return Data.param_ranges[param]
        else:
            return None

    def __str__(self):
        return (self.xparam + "," + self.yparam + "\n" + str(self.xvals) + "\n" + str(self.yvals))

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


def get_avg_data(infile, x_params, y_params, sport_types):
    """
    returns data averaged over each workout for each sport type. For each i, returns y_params[i] and x_params[i] averaged over each workout. Thus each point in the plot corresponds to one workout
    x_params  list of parameters, must be present in the data
    y_params  list of parameters, must be present in the data
    Must be true : len(x_params) == len(y_params)
    """
    assert(len(x_params) == len(y_params))
    assert(len(x_params) == len(sport_types))
    assert(len(x_params) > 0)
    assert(len(y_params) > 0)
    n_params = len(x_params)
    #assert(sport in ["Running", "Cycling", "Walking", "Circuit Training", "Mountain biking"])

    # create lists to store data objects
    objs = []
    #for s in sport_types:
        #objs[s] = []
    for i in range(0, n_params):
        objs.append(Data(xparam = x_params[i], yparam = y_params[i], sport = sport_types[i]))

    print "X parameters : " + str(x_params)
    print "Y parameters : " + str(y_params)
    print "Sports : " + str(sport_types)

    infile_basename, ext = os.path.splitext(infile)
    if (ext == ".gz"):
        f = gzip.open(infile)
    elif(ext == ".txt"):
        f = open(infile)
    else:
        raise Exception("File format not recognized")

    n_lines = 0
    for line in f:
        # each line is a workout
        w = utils.json_to_dict(line.strip())
        sport = w["sport"]
        for i in range(0, n_params):
            if (sport != sport_types[i]):
                continue
            xp = x_params[i]; yp = y_params[i]      # x and y axis parameters
            if (w.has_key(xp) and w.has_key(yp)):
                [x_trace, y_trace] = utils.remove_null_values(w[xp], w[yp])
                mx = np.mean(x_trace)
                my = np.mean(y_trace)
                objs[i].add_point(mx, my)
        n_lines += 1

    f.close()
    
    return objs


def plot_avg_data(x, y, x_params, y_params):
    assert(len(x_params) == len(y_params))
    for i in range(len(x_params)):
        plt.figure(i)
        plt.plot(x[i], y[i])
        plt.xlabel(x_params[i])
        plt.ylabel(y_params[i])
    plt.show()

def print_summary(data_objs):
    pass

def visualize_all(infile):
    x_params = ["alt", "hr", "duration", "distance"] * 2
    y_params = ["pace"] * 4 + ["speed"] * 4
    #y_params = ["speed", "speed", "speed", "speed"]
    sports = ["Running"] * 4 + ["Cycling, sport"] * 4
    objs = get_avg_data(infile, x_params, y_params, sports)
    print Data.summary_format()
    for i in range(0, len(x_params)):
        d = objs[i]
        if (not d.empty()):
            d.plot()
        print d.summary()
    plt.show()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='See various plots over all workouts')
    parser.add_argument('--infile', type=str, help='.gz file or .txt containing all workouts', dest='infile')
    args = parser.parse_args()
    if (args.infile is None):
        parser.print_usage()
        exit(0)
    visualize_all(args.infile)

