from matplotlib import pyplot as plt
import numpy as np
import csv

data = []
with open("data_overfitting.dat") as f:
    reader = csv.reader(f)
    for row in reader:
        data.append(np.asarray(row, dtype=np.float))
data = list(zip(*data))

plt.plot(data[0],data[1], label="training set")
plt.plot(data[0],data[2], label="validation set")
plt.xlabel("m")
plt.ylabel("error %")
plt.grid()
plt.legend()

plt.savefig("graph_errs_overfitting.png")
plt.show()


