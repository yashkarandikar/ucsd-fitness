import os
import time

#os.system("python predictor_duration_evolving_user.py 0.0 0.0 > output_0_0.txt")
t1 = time.time()
lam1 = 0.0
while (lam1 <= 1.0):
    lam2 = 0.0
    while (lam2 <= 1.0):
        ofile = "runs_evolving_model_hr/final_E2_10/output_%f_%f.txt" % (lam1, lam2)
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
