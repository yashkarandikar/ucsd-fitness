import matplotlib.pyplot as plt

def main():
    infile = "output.txt"
    e = []
    lam = 0.0
    r2 = []
    with open(infile) as f:
        for line in f:
            if (line.startswith('@')):
                line = line.strip()[1:]
                e.append(line.split("=")[1].strip())
            elif (line.startswith("R2")):
                parts = line.strip().split("=")
                r2.append(float(parts[2].strip()))
            elif (line.startswith("lambda")):
                parts = line.strip().split("=")
                lam = float(parts[1].strip())
    plt.plot(range(0, len(e)), e)
    plt.xlabel("Iteration number")
    plt.ylabel("Error")
    plt.title("E = %d, lambda = %f, Training R2 = %f, Validation %f" % (3, lam, r2[0], r2[1]))
    plt.show()

if  __name__ == "__main__":
    main()
