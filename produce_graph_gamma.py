import csv
import collections
import numpy as np
from matplotlib import pyplot as plt

with open("data_gamma_n0.dat") as f:
    reader = csv.reader(f)
    i = 0
    gammas = []
    thetas = []
    for r in reader:
        if i%2==0:
            gammas.append(np.float(r[0]))
        else:
            thetas.append(np.asarray(r,dtype=np.float))
        i+=1

ys = list(zip(*thetas))
for i in ys:
    plt.plot(gammas, i)
plt.show()

