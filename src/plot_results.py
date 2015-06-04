import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def plot(labels, train, val, test):
    r2 = np.matrix([train, val, test]).T
    fig, ax = plt.subplots()
    width = 0.20
    colors = ["#78EC2B", "#2299FF", "#AB0000"]
    nrows = r2.shape[0]
    x = np.arange(0, 3)
    for i in range(0, nrows):
        vals = r2[i,:].A1
        ax.bar(x+i*width, vals, width, label = labels[i], color = colors[i], edgecolor = "none")
    ax.set_xticks(x+(width * nrows) / 2.0)
    ax.set_xticklabels(('G1', 'G2', 'G3'))
    ax.set_ylim([0, 1])
    plt.legend()

def plot_lines(lines):
    for i in xrange(1, 4):
        parts = lines[i].strip().split(",")
        labels = parts[0]
        train = float(parts[1])
        val = float(parts[2])
        test = float(parts[3])

if __name__ == "__main__":
    labels = ["Baseline", "E=2", "E=3"]
    train = [0.5, 0.6, 0.7]
    val = [0.4, 0.4, 0.4]
    test = [0.2, 0.2, 0.2]
    plot(labels, train, val, test)
    plt.show()
