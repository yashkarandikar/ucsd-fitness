import os
import time

t1 = time.time()
lam1 = 0.0
lam2 = 0.0
while (lam2 <= 1.0):
    ofile = "runs_model2_inst/random_E1/output_%f.txt" % (lam2)
    command = "python predictor_insthr_evolving.py %f %f 1 1 > %s" % (lam1, lam2, ofile)
    print "Running : ", command
    os.system(command)
    if (lam2 == 0.0):
        lam2 = 0.001
    else:
        lam2 = lam2 * 10.0
print "Done"
t2 = time.time()
lam1 = 1.0
lam2 = 0.1
ofile = "runs_model2_inst/random_E1/output_%f_%f.txt" % (lam1, lam2)
command = "python predictor_insthr_evolving.py %f %f 1 1 > %s" % (lam1, lam2, ofile)
print "Running : ", command
os.system(command)
print "Total time taken  = ", t2 - t1
