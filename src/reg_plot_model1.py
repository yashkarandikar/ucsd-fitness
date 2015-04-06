import matplotlib.pyplot as plt
import sys

def main():
    files = sys.argv[1:]
    train = {}
    val = {}
    found = {'train_r2' : False, 'val_r2' : False, 'lam1' : False}
    for fi in files:
        with open(fi) as f:
            train_r2 = 0.0; val_r2 = 0.0; lam1 = 0.0
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
                    if ("lam1" in line):
                        print line
                        parts = line.split(",")
                        for p in parts:
                            k, v = p.split("=")
                            k = k.strip()
                            v = v.strip()
                            if (k == "lam1"):
                                lam1 = float(v)
                                found['lam1'] = True
            found_all = True
            for v in found.values():
                if (not v):
                    found_all = False
                    break
            if (found_all):
                if (lam1 == 0):
                    lam1 = 1e-12;
                train[lam1] = train_r2
                val[lam1] = val_r2

    print train
    print val

    max_r2 = 0.0
    max_lam1 = 0.0
    plt.figure()
    
    lam_list = []
    train_r2_list = []
    for k1, v1 in sorted(train.items()):
        lam1 = k1
        r2 = v1
        lam_list.append(lam1)
        train_r2_list.append(r2)
    both = zip(lam_list, train_r2_list)
    both.sort()
    lam_list = [a for a,b in both]
    train_r2_list = [b for a,b in both]
    plt.plot(lam_list, train_r2_list, label = "Training", marker = "o")
    
    lam_list = []
    val_r2_list = []
    for k1, v1 in sorted(val.items()):
        lam1 = k1
        r2 = v1
        lam_list.append(lam1)
        val_r2_list.append(r2)
        if (r2 > max_r2):
            max_r2 = r2
            max_lam1 = lam1

    both = zip(lam_list, val_r2_list)
    both.sort()
    lam_list = [a for a,b in both]
    val_r2_list = [b for a,b in both]
    plt.plot(lam_list, val_r2_list, label = "Validation", marker = "o")
    print "Best Validation R2 = %f is at lam1 = %f" % (max_r2, max_lam1)
    print "Training R2 at lam1 = %f is %f" % (max_lam1, train[max_lam1])
    plt.xlabel("lambda")
    plt.ylabel("R2 on validation set")
    plt.legend(loc = "best")
    plt.semilogx(basex=10)
    #plt.xscale('symlog')
    plt.show()

if __name__ == "__main__":
    main()
