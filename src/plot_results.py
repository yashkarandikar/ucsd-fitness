import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def plot(labels, train, val, test):
    r2 = np.matrix([train, val, test]).T
    fig, ax = plt.subplots()
    width = 0.20
    colors = ["#3465ff", "#fbbf2a", "#de2d0b"]
    nrows = r2.shape[0]
    x = np.arange(1, 4)
    for i in range(0, nrows):
        vals = r2[i,:].A1
        ax.bar(x+i*width, vals, width, label = labels[i], color = colors[i], edgecolor = "none")
    ax.set_xticks(x+(width * nrows) / 2.0)
    ax.set_xticklabels(('Training', 'Validation', 'Test'))
    ax.set_ylabel(r'Coefficient of Determination $R^2$')
    ax.set_ylim([0, 1])
    plt.legend(loc = "best")

def plot_lines(lines):
    labels = []
    train = []
    val = []
    test = []
    for i in xrange(1, 4):
        parts = lines[i].strip().split(",")
        labels.append(parts[0])
        train.append(float(parts[1]))
        val.append(float(parts[2]))
        test.append(float(parts[3]))
    plot(labels, train, val, test)

def plot_file(filename):
    #labels = ["Baseline", "E=2", "E=3"]
    #train = [0.5, 0.6, 0.7]
    #val = [0.4, 0.4, 0.4]
    #test = [0.2, 0.2, 0.2]
    #plot(labels, train, val, test)
    with open(filename) as f:
        lines = f.readlines()
        plot_lines(lines)
    plt.show()
    #plt.savefig(filename + ".png")

if __name__ == "__main__":
    plot_file("results_duration_final.csv")
    plot_file("results_duration_random.csv")
    plot_file("results_hr_final.csv")
    plot_file("results_hr_random.csv")
