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
                        train_r2 = float(line.split("=")[-1].strip())
                        found['train_r2'] =  True
                    elif (line.startswith("Validation")):
                        val_r2 = float(line.split("=")[-1].strip())
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
                train[lam1][lam2] = train_r2
                val[lam1][lam2] = val_r2

    print train
    print val

    plt.figure()
    for k1, v1 in val.items():
        lam1 = k1
        lam2 = []; r2 = []
        for k2, v2 in v1.items():
            lam2.append(k2)
            r2.append(v2)
        both = zip(lam2, r2)
        both.sort()
        lam2 = [a for a,b in both]
        r2 = [b for a,b in both]
        print lam2, r2
        plt.plot(lam2, r2, label = "Lam1 = %f" % (lam1))
    plt.legend()
    plt.semilogx(basex=10)
    plt.show()

if __name__ == "__main__":
    main()
