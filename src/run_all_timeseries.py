import time
import os

t1 = time.time()
#E = [50, 1, 10, 20, 30, 40]
E = [30]
lam = [1000.0, 10000.0, 100000.0]
output_dir = "runs_timeseries/new/E/30"
for e in E:
    for l in lam:
        ofile = output_dir + "/" + "output_" + str(l) + "_" + str(e) + ".txt"
        command = "python timeseries_linear.py %f %d > %s" % (l, e, ofile)
        print "Running : ", command
        os.system(command)
t2 = time.time()
