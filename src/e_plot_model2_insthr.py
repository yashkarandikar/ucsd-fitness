import matplotlib.pyplot as plt
import sys

def main():
    files = sys.argv[1:]
    train = {}
    val = {}
    found = {'train_r2' : False, 'val_r2' : False, 'lam1' : False, 'lam2' : False}
    for fi in files:
        with open(fi) as f:
            train_r2 = 0.0; val_r2 = 0.0; lam1 = 0.0; lam2 = 0.0
            for k in found.keys():
                found[k] = False
            for line in f:
                if (line[0] == '@'):
                    line = line[1:]
                    if (line.startswith("Training")):
                        train_r2 = float(line.split(",")[-2].split("=")[2].strip())
                        e = float(line.split(",")[-1].split("=")[1].strip())
                        train[e] = train_r2
                    elif (line.startswith("Validation")):
                        val_r2 = float(line.split(",")[-2].split("=")[2].strip())
                        e = float(line.split(",")[-1].split("=")[1].strip())
                        val[e] = val_r2

    print train
    print val

    plt.figure()
    train_r2 = []
    E = []
    for k1, v1 in sorted(train.items()):
        E.append(k1)
        train_r2.append(v1)
    print E
    print train_r2
    plt.plot(E, train_r2, marker = "o", label = "Train R2")
    
    val_r2 = []
    E = []
    for k1, v1 in sorted(val.items()):
        E.append(k1)
        val_r2.append(v1)
    plt.plot(E, val_r2, marker = "o", label = "Validation R2")

    plt.xlabel("Number of tiredness levels")
    plt.ylabel("R^2")
    plt.title("Prediction of instantaneous HR")
    plt.legend(loc = "best")

    plt.show()
    
if __name__ == "__main__":
    main()
