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
import math

class DataForPlot(object):

    def __init__(self, sport, xparam, yparam, xvals = None, yvals = None):
        self.xparam = xparam
        self.yparam = yparam
        self.sport = sport
        if (xvals is None):
            xvals = []
        if (yvals is None):
            yvals = []
        self.xvals = xvals
        self.yvals = yvals
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

    def plot_simple(self, x_range = None, y_range = None):
        assert(len(self.xvals) == len(self.yvals))
        self.sort()
        #plt.figure()
        plt.plot(self.xvals, self.yvals, 'o')
        if (x_range is not None):
            plt.xlim(x_range)
        if (y_range is not None):
            plt.ylim(y_range)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.describe())

    def describe(self):
        return self.sport + " : " + self.yparam + " vs " + self.xparam

    def plot_bar(self):
        assert(len(self.xvals) == len(self.yvals))
        self.sort()
        barplot.barplot(self.xvals, self.yvals, nbins=10, xparam=self.xlabel, yparam=self.ylabel, title="Sport: %s"%(self.sport))

    def plot_sliding(self, windowSize, x_range = None, y_range = None):
        assert(len(self.xvals) == len(self.yvals))
        n = len(self.xvals)
        assert(n > windowSize)
        self.sort()
        #plt.figure()
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
        plt.plot(x, y, 'o')
        #xlims = self.get_range(param_ranges, self.xparam)
        #ylims = self.get_range(param_ranges, self.yparam)
        #if (xlims is not None and max(x) > max(xlims)):
            #plt.xlim(xlims)
        #if (ylims is not None and max(y) > max(ylims)):
            #plt.ylim(ylims)
        if (x_range is not None):
            plt.xlim(x_range)
        if (y_range is not None):
            plt.ylim(y_range)

        plt.xlabel(self.xlabel + "(window %d)" % (windowSize))
        plt.ylabel(self.ylabel + "(window %d)" % (windowSize))
        plt.title(self.describe() + "(window %d)" % (windowSize))

    def plot_sliding_Y(self, windowSize, x_range = None, y_range = None):
        assert(len(self.xvals) == len(self.yvals))
        n = len(self.xvals)
        assert(n > windowSize)
        self.sort()
        plt.figure()
        x = [0] * (n - windowSize + 1)
        y = [0] * (n - windowSize + 1)
        w = float(windowSize)
        sum_y = float(sum(self.yvals[0:windowSize]))
        avg_y = sum_y / w
        w_mid = int(math.floor(windowSize / 2.0))
        x[0] = self.xvals[w_mid]
        y[0] = avg_y
        j = 1
        for i in range(1 + w_mid, n - w_mid):
            x[j] = self.xvals[i]
            sum_y = sum_y - self.yvals[j-1] + self.yvals[j + windowSize - 1]
            y[j] = sum_y / w
            j += 1
        plt.plot(x, y, 'o')
        #xlims = self.get_range(param_ranges, self.xparam)
        #ylims = self.get_range(param_ranges, self.yparam)
        #if (xlims is not None and max(x) > max(xlims)):
            #plt.xlim(xlims)
        #if (ylims is not None and max(y) > max(ylims)):
            #plt.ylim(ylims)
        if (x_range is not None):
            plt.xlim(x_range)
        if (y_range is not None):
            plt.ylim(y_range)

        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel + "(averaged over window size %d)" % (windowSize))
        plt.title(self.describe())


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
    For each i, returns y_params[i] and x_params[i] where sport = sport_types[i]. Thus each point in the plot corresponds to one workout
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
        objs.append(DataForPlot(xparam = x_params[i], yparam = y_params[i], sport = sport_types[i]))

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
                objs[i].add_point(mx, my)
        nw += 1
        if (nw % 100000 == 0):
            print "Done processing %s workouts" % (nw)

    f.close()
    
    return objs


def plot_data(infile, x_params, y_params, sports, x_ranges, y_ranges, windowSize = 100, use_saved = False):
    t1 = time.time()
    assert(len(x_params) == len(y_params) and len(x_params) == len(x_ranges) and len(x_ranges) == len(y_ranges))
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
    print DataForPlot.summary_format()

    nplots = len(x_params)
    ncols = 2
    nrows = int(math.ceil(nplots / 2.0))

    plt.figure(1)
    for i in range(0, len(x_params)):
        d = objs[i]
        plt.subplot(nrows,ncols,i)
        if (not d.empty()):
            d.plot_sliding(windowSize = windowSize, x_range = x_ranges[i], y_range = y_ranges[i])
            #d.plot_sliding_Y(windowSize=100, x_range = x_ranges[i], y_range = y_ranges[i])
            #d.plot_simple(x_range = x_ranges[i], y_range = y_ranges[i])
        else:
            print d.describe() + " is empty.."
        print d.summary()

    t2 = time.time()
    print "Time taken = " + str(t2 - t1)

    plt.show()

def plot_duration_vs_all(infile, use_saved):
    x_params = ["pace(avg)", "alt(avg)", "hr(avg)", "Distance", "Total Ascent", "Total Descent"]
    y_params = ["Duration"] * 6
    sports = ["Running"] * 6
    x_ranges = [[0, 25], [0, 10000], [0, 225], [0, 50], [0, 30000], [0, 30000]]
    y_ranges = [[0, 30000], [0, 30000], [0, 10000], [0, 80000],[0, 80000], [0, 80000]]
    assert(len(x_params) == len(y_params) and len(x_params) == len(x_ranges) and len(x_ranges) == len(y_ranges))
    plot_data(infile, x_params, y_params, sports, x_ranges, y_ranges, use_saved = use_saved)

def plot_hr_vs_all(infile, use_saved):
    x_params = ["pace(avg)", "alt(avg)", "Duration", "Distance", "Total Ascent", "Total Descent"]
    y_params = ["hr(avg)"] * 6
    sports = ["Running"] * 6
    x_ranges = [[0, 20],[0, 10000],[0, 30000],[0, 50],[0, 20000],[0, 20000]]
    y_ranges = [[50, 200]] * 6
    assert(len(x_params) == len(y_params) and len(x_params) == len(x_ranges) and len(x_ranges) == len(y_ranges))
    plot_data(infile, x_params, y_params, sports, x_ranges, y_ranges, use_saved = use_saved, windowSize = 100)

def plot_calories_vs_all(infile, use_saved):
    x_params = ["pace(avg)", "alt(avg)", "Duration", "Distance", "hr(avg)"]
    y_params = ["Calories"] * 5
    sports = ["Running"] * 5
    x_ranges = [[0, 20],[0, 10000],[0, 30000],[0, 50], [50, 200]]
    y_ranges = [[0, 3000], [0, 2000], [0, 4000], [0, 6000], [0, 1100]]
    assert(len(x_params) == len(y_params) and len(x_params) == len(x_ranges) and len(x_ranges) == len(y_ranges))
    plot_data(infile, x_params, y_params, sports, x_ranges, y_ranges, use_saved = use_saved, windowSize = 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='See various plots over all workouts')
    parser.add_argument('--infile', type=str, help='.gz file or .txt containing all workouts', dest='infile')
    parser.add_argument('--use-saved', action='store_true', help='generate plots from previously saved data objects', default=False, dest='use_saved')
    args = parser.parse_args()
    if (args.infile is None):
        parser.print_usage()
        exit(0)
    #plot_duration_vs_all(args.infile, args.use_saved)
    #plot_hr_vs_all(args.infile, args.use_saved)
    plot_calories_vs_all(args.infile, args.use_saved)

