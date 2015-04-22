import os
import time

if __name__ == "__main__":
    t1 = time.time()
    for e in range(1, 40, 3):
        command = "python predictor_insthr_evolving.py 0.1 0.001 %d > runs_model2_inst/e%d.txt" % (e, e)
        print "Running : ", command
        os.system(command)
    t2 = time.time()
    print "Total time = ", t2 - t1
