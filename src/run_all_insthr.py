import os
import time

t1 = time.time()
lam1 = 0.0
while (lam1 <= 1.0):
    lam2 = 0.0
    while (lam2 <= 1.0):
        ofile = "runs_model2_inst/final_E10/output_%f_%f.txt" % (lam1, lam2)
        command = "python predictor_insthr_evolving.py %f %f 10 > %s" % (lam1, lam2, ofile)
        print "Running : ", command
        os.system(command)
        if (lam2 == 0.0):
            lam2 = 0.001
        else:
            lam2 = lam2 * 10.0
    if (lam1 == 0.0):
        lam1 = 0.001
    else:
        lam1 = lam1 * 10.0
print "Done"
t2 = time.time()
print "Total time taken  = ", t2 - t1
