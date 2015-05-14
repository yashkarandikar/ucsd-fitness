import os
import time

l1 = [0.0, 0.0001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
t1 = time.time()
for lam1 in l1:
    if (lam1 >= 100.0):
        ofile = "runs_model1_hr/random_E1_10/output_%f.txt" % (lam1)
        command = "python predictor_avghr_user.py %f > %s" % (lam1, ofile)
        print "Running : ", command
        os.system(command)
t2 = time.time()
print "Total time taken  = ", t2 - t1

"""
t1 = time.time()
lam1 = 0.0
while (lam1 <= 10.0):
    #ofile = "runs_model1/final_20/output_%f.txt" % (lam1)
    ofile = "runs_model1_hr/random_E1_10/output_%f.txt" % (lam1)
    command = "python predictor_avghr_user.py %f > %s" % (lam1, ofile)
    print "Running : ", command
    os.system(command)
    if (lam1 == 0.0):
        lam1 = 0.0001
    else:
        lam1 = lam1 * 10.0
print "Done"
t2 = time.time()
print "Total time taken  = ", t2 - t1
"""
