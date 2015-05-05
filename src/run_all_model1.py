import os
import time

t1 = time.time()
lam1 = 0.0
while (lam1 <= 10.0):
    #ofile = "runs_model1/final_20/output_%f.txt" % (lam1)
    ofile = "runs_model1/random_E1_10/output_%f.txt" % (lam1)
    command = "python predictor_duration_user.py %f > %s" % (lam1, ofile)
    print "Running : ", command
    os.system(command)
    if (lam1 == 0.0):
        lam1 = 0.0001
    else:
        lam1 = lam1 * 10.0
print "Done"
t2 = time.time()
print "Total time taken  = ", t2 - t1
