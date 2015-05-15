import os
import time

#l1 = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
#l2 = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
l1 = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
l2 = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
t1 = time.time()
for lam1 in l1:
    for lam2 in l2:
        if (lam1 >= 100.0 or lam2 >= 100.0):
            ofile = "runs_evolving_model/final_E3_10/output_%f_%f.txt" % (lam1, lam2)
            command = "python predictor_duration_evolving_user.py %f %f > %s" % (lam1, lam2, ofile)
            print "Running : ", command
            os.system(command)
t2 = time.time()
print "Total time taken  = ", t2 - t1

#os.system("python predictor_duration_evolving_user.py 0.0 0.0 > output_0_0.txt")
"""
t1 = time.time()
lam1 = 0.0
while (lam1 <= 10.0):
    lam2 = 0.0
    while (lam2 <= 10.0):
        ofile = "runs_evolving_model_hr/final_E3_10/output_%f_%f.txt" % (lam1, lam2)
        command = "python predictor_avghr_evolving_user.py %f %f > %s" % (lam1, lam2, ofile)
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
"""
