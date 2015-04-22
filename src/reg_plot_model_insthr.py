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
                        #train_r2 = float(line.split("=")[-1].strip())
                        train_r2 = float(line.split(",")[-2].split("=")[2].strip())
                        found['train_r2'] =  True
                    elif (line.startswith("Validation")):
                        #val_r2 = float(line.split("=")[-1].strip())
                        val_r2 = float(line.split(",")[-2].split("=")[2].strip())
                        found['val_r2'] = True
                    if ("lam1" in line and "lam2" in line):
                        parts = line.split(",")
                        for p in parts:
                            k, v = p.split("=")
                            k = k.strip()
                            v = v.strip()
                            if (k == "lam1"):
                                lam1 = float(v)
                                found['lam1'] = True
                            if (k == "lam2"):
                                lam2 = float(v)
                                found['lam2'] = True
            found_all = True
            for v in found.values():
                if (not v):
                    found_all = False
                    break
            if (found_all):
                if (not train.has_key(lam1)):
                    train[lam1] = {}
                if (not val.has_key(lam1)):
                    val[lam1] = {}
                if (lam2 == 0):
                    lam2 = 1e-12;
                train[lam1][lam2] = train_r2
                val[lam1][lam2] = val_r2

    print train
    print val

    max_r2 = 0.0
    max_lam1 = 0.0
    max_lam2 = 0.0
    plt.figure()
    for k1, v1 in sorted(val.items()):
        lam1 = k1
        lam2 = []; r2 = []
        for k2, v2 in v1.items():
            lam2.append(k2)
            r2.append(v2)
            if (v2 > max_r2):
                max_r2 = v2
                max_lam1 = lam1
                max_lam2 = k2

        both = zip(lam2, r2)
        both.sort()
        lam2 = [a for a,b in both]
        r2 = [b for a,b in both]
        print "lam1 = " + str(lam1) + ", " + str(lam2) + ", " + str(r2)
        plt.plot(lam2, r2, label = "Lam1 = %1.0e" % (lam1), marker = "o")
    print "Best Validation R2 = %f is at lam1 = %f and lam2 = %f" % (max_r2, max_lam1, max_lam2)
    print "Training R2 at lam1 = %f and lam2 = %f is %f" % (max_lam1, max_lam2, train[max_lam1][max_lam2])
    plt.xlabel("lam2")
    plt.ylabel("R2 on validation set")
    plt.legend(loc = "best")
    plt.semilogx(basex=10)
    plt.show()

if __name__ == "__main__":
    main()
