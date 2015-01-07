#!/usr/bin/python

import argparse
import gzip
import numpy as np
import utils
import matplotlib.pyplot as plt
import os 
import barplot
import time
import pickle
from unit import Unit

class Data(object):

    def __init__(self, sport, xparam, yparam):
        self.xparam = xparam
        self.yparam = yparam
        self.sport = sport
        self.xvals = []
        self.yvals = []
        self.xlabel = self.xparam + " (%s)"%(Unit.get(self.xparam))
        self.ylabel = self.yparam + " (%s)"%(Unit.get(self.yparam))

    def add_point(self, x, y):
        self.xvals.append(x)
        self.yvals.append(y)

    def sort(self):
        xy = zip(self.xvals, self.yvals)
        xy.sort()
        self.xvals = [a for a, b in xy]
        self.yvals = [b for a, b in xy]

    """
    def plot(self):
        assert(len(self.xvals) == len(self.yvals))
        self.sort()
        plt.figure()
        plt.plot(self.xvals, self.yvals)
        xlims = Data.param_range(self.xparam)
        ylims = Data.param_range(self.yparam)
        if (xlims is not None):
            print "Using range " + xlims + " for " + self.xparam
            plt.xlim(xlims)
        if (ylims is not None):
            print "Using range " + ylims + " for " + self.yparam
            plt.ylim(ylims)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title("Sport: %s"%(self.sport))
    """

    def describe(self):
        return self.sport + " : " + self.yparam + " vs " + self.xparam

    def plot_bar(self):
        assert(len(self.xvals) == len(self.yvals))
        self.sort()
        barplot.barplot(self.xvals, self.yvals, nbins=10, xparam=self.xlabel, yparam=self.ylabel, title="Sport: %s"%(self.sport))

    def plot_sliding(self, windowSize, param_ranges = None):
        assert(len(self.xvals) == len(self.yvals))
        n = len(self.xvals)
        assert(n > windowSize)
        self.sort()
        plt.figure()
        x = [0] * (n - windowSize + 1)
        y = [0] * (n - windowSize + 1)
        w = float(windowSize)
        sum_x = float(sum(self.xvals[0:windowSize]))
        sum_y = float(sum(self.yvals[0:windowSize]))
        avg_x = sum_x / w
        avg_y = sum_y / w
        x[0] = avg_x
        y[0] = avg_y
        for i in range(1, n - windowSize + 1):
            sum_x = sum_x - self.xvals[i-1] + self.xvals[i + windowSize - 1]
            sum_y = sum_y - self.yvals[i-1] + self.yvals[i + windowSize - 1]
            avg_x = sum_x / w
            avg_y = sum_y / w
            x[i] = avg_x
            y[i] = avg_y
        plt.plot(x, y)
        xlims = self.get_range(param_ranges, self.xparam)
        ylims = self.get_range(param_ranges, self.yparam)
        if (xlims is not None and max(x) > max(xlims)):
            plt.xlim(xlims)
        if (ylims is not None and max(y) > max(ylims)):
            plt.ylim(ylims)

        plt.xlabel(self.xlabel + "(averaged over window size %d)" % (windowSize))
        plt.ylabel(self.ylabel + "(averaged over window size %d)" % (windowSize))
        plt.title(self.describe() + "(averaged over window size %d)" % (windowSize))

    def empty(self):
        return (len(self.xvals) == 0)

    def summary(self):
        return "{:<15}{:<15}{:<15}{:<15}".format(self.sport, self.xparam, self.yparam, len(self.xvals))

    @staticmethod
    def summary_format():
        return "{:<15}{:<15}{:<15}{:<15}".format("Sport","X param","Y param","# workouts")

    def get_range(self, param_ranges, param):
        if (param_ranges is not None and param_ranges.has_key(param)):
            return param_ranges[param]
        else:
            return None

    def __str__(self):
        return (self.xparam + "," + self.yparam + "\n" + str(self.xvals) + "\n" + str(self.yvals))

def get_data(infile, x_params, y_params, sport_types):
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

    nw = 0
    for line in f:
        # each line is a workout
        w = utils.json_to_dict(line.strip())
        sport = w["sport"]
        for i in range(0, n_params):
            if (sport != sport_types[i]):
                continue
            xp = x_params[i]; yp = y_params[i]      # x and y axis parameters
            if (w.has_key(xp) and w.has_key(yp)):
                mx = w[xp]
                my = w[yp]
                #if (isinstance(mx, list) and isinstance(my, list)):
                #    [x_trace, y_trace] = utils.remove_null_values(mx, my)
                #    mx = np.mean(x_trace)
                #    my = np.mean(y_trace)
                #elif (isinstance(mx, list)):
                #    x_trace = utils.remove_null_values_single(mx)
                #    mx = np.mean(x_trace)
                #elif (isinstance(my, list)):
                #    y_trace = utils.remove_null_values_single(my)
                #    my = np.mean(y_trace)
                objs[i].add_point(mx, my)
        nw += 1
        if (nw % 10000 == 0):
            print "Done processing %s workouts" % (nw)

    f.close()
    
    return objs


def plot_all(infile, use_saved = False):
    t1 = time.time()
    #x_params = ["alt", "hr", "duration", "distance"] * 2
    #y_params = ["pace"] * 4 + ["speed"] * 4
    #y_params = ["speed", "speed", "speed", "speed"]
    #sports = ["Running"] * 4 + ["Cycling, sport"] * 4
    x_params = ["pace(avg)", "alt(avg)", "hr(avg)", "Distance"]
    y_params = ["Duration"] * 4
    sports = ["Running"] * 4
    param_ranges = {"Duration" : [0, 50000], "Distance" : [0, 40], "pace(avg)" : [0, 40], "hr(avg)":[0, 300], "alt(avg)" : [0,10000]}
    fName, fExt = os.path.splitext(infile)
    plot_outfile = fName + "_plots.txt"
    if (use_saved):
        with open(plot_outfile, 'r') as f:
            objs_str = f.read()
        objs = pickle.loads(objs_str)
        print "Plot data read from previously saved file " + plot_outfile
    else:
        objs = get_data(infile, x_params, y_params, sports)
        objs_str = pickle.dumps(objs)
        with open(plot_outfile, 'w') as f:
            f.write(objs_str)
        print "Plot data stored in " + plot_outfile
    print Data.summary_format()

    for i in range(0, len(x_params)):
        d = objs[i]
        if (not d.empty()):
            #d.plot()
            #d.plot_bar()
            #print d
            d.plot_sliding(windowSize=100, param_ranges = param_ranges)
        else:
            print d.describe() + " is empty.."
        print d.summary()

    t2 = time.time()
    print "Time taken = " + str(t2 - t1)
    
    plt.show()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='See various plots over all workouts')
    parser.add_argument('--infile', type=str, help='.gz file or .txt containing all workouts', dest='infile')
    parser.add_argument('--use-saved', action='store_true', help='generate plots from previously saved data objects', default=False, dest='use_saved')
    args = parser.parse_args()
    if (args.infile is None):
        parser.print_usage()
        exit(0)
    plot_all(args.infile, args.use_saved)

