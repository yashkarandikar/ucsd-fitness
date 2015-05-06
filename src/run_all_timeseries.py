import time
import os

t1 = time.time()
E = [50, 1, 10, 20, 30, 40]
lam = 10000.0
output_dir = "runs_timeseries/new/E"
for e in E:
    ofile = output_dir + "/" + "output_" + str(e) + ".txt"
    command = "python timeseries_linear.py %f %d > %s" % (lam, e, ofile)
    print "Running : ", command
    os.system(command)
t2 = time.time()
