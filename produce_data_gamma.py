import HW5lib as HW
import numpy as np
import csv
import collections

imgs_train = np.matrix(np.r_[HW._mnst.train.images])
imgs_test  = np.matrix(np.r_[HW._mnst.test.images])
lbls_train = np.matrix(HW._mnst.train.labels).T
lbls_test  = np.matrix(HW._mnst.test.labels).T

m = 500
gammas = np.arange(0.001,0.01,0.001)

thetas = collections.defaultdict(lambda: [])

for g in gammas:
    print("Creation network g={}".format(g))
    net = HW.mnist_classifier(m,imgs_train,lbls_train,g,True)
    for i in range(10):
        thetas[i].append(net.nets[i]._theta.A1)

for i in range(10):
    with open("data_gamma_n{}.dat".format(i), "w") as f:
        writer = csv.writer(f)
        for j,g in enumerate(gammas):
            writer.writerow([g])
            writer.writerow(thetas[i][j])

