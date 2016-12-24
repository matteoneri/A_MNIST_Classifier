from HW5lib_profile import *

net = mnist_classifier(1000, imgs, lbls)

#net.compute_error(imgs,lbls)
net.compute_error(imgs_val,lbls_val)
