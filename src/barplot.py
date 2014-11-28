import matplotlib.pyplot as plt
import random
import numpy
import math

def barplot(x, y, nbins, xparam="", yparam="", title=""):
    fig, ax = plt.subplots()
    min_x = min(x)
    max_x = max(x)
    bin_size = math.ceil(float(max_x - min_x) / float(nbins))
    bins = [min_x + i*bin_size for i in range(nbins)]
    bins_index = numpy.searchsorted(x, bins)
    left = []
    height = []
    labels = []
    xticks = []
    for i in range(0, nbins):
        s = bins[i]
        si = bins_index[i]
        left.append(s)
        if i < nbins - 1:
            ei = bins_index[i + 1]
        else:
            ei = len(y)
        xticks.append(s)
        y_part = y[si:ei]
        if (len(y_part) > 0):
            height.append(numpy.mean(y_part))
        else:
            height.append(0)
        labels.append("%d"%(s))
    ax.bar(left = left, height = height, width = bin_size)
    ax.set_xticklabels(labels)
    ax.set_xticks(xticks)
    ax.set_xlim(min_x, max_x)
    ax.set_xlabel(xparam)
    ax.set_ylabel(yparam)
    ax.set_title(title)
    return ax
    #plt.show()

if __name__ == "__main__":
    alt = [0, 10, 20, 30, 40, 50, 60, 70]
    pace = [100, 102, 106, 110, 110, 110, 115, 120]
    barplot(alt, pace, 3, xparam="alt", yparam="pace")
    plt.show()
