from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from scipy import linalg
from scipy import sparse
from functools import reduce

_mnst  = input_data.read_data_sets("../MNIST_data/", validation_size=10000)
imgs   = np.matrix(np.r_[_mnst.train.images,_mnst.test.images])
lbls   = np.matrix(np.r_[_mnst.train.labels,_mnst.test.labels]).T

imgs_val = np.matrix(np.r_[_mnst.validation.images])
lbls_val = np.matrix(_mnst.validation.labels).T

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(_R = np.matrix([]))
class neural_net:
    def __init__(self, m, n, gamma = 0.1, static_matrix = False, sparse = False):
        self._m      = m
        self._n      = n
        if static_matrix and not neural_net._R.any():
            np.random.seed(3141592653)
            neural_net._R = np.matrix(np.random.randn(n,m), dtype=np.float32)
            print("STATIC")
        if not static_matrix:
            self._R      = np.matrix(np.random.randn(n,m), dtype=np.float32)
            print("NO STATIC")
        self._theta  = np.matrix(np.ones(m, dtype=np.float32))
        self._gamma  = np.float32(gamma)
        self._sparse = sparse

    @staticmethod
    def sigmoid(z):
        #return 1/(1+np.exp(-z))
        return np.reciprocal(np.add(1,np.exp(-z,z),z),z)

    def predict(self,x):
        return (self.sigmoid(x*self._R)*self._theta).A1

    def train(self,x,y):
        z = neural_net.sigmoid(x*self._R)
        self._theta = np.matrix(linalg.solve(z.T*z+self._gamma*np.eye(self._m, dtype=np.float32), z.T*y))
        #self._theta = (z.T*z+self._gamma*np.eye(self._m, dtype=np.float32)).I * z.T*y
        return "done"



class mnist_classifier:
    @staticmethod
    def y(lbls,i):
        return (lbls == i)*np.float32(2)-np.float32(1)

    def __init__(self,m,imgs,lbls,gamma=0.1,static_matrix=False):
        self.nets = [neural_net(m,28**2,gamma,static_matrix) for l in range(10)]
        print("training [{:<10}] {}%".format("", 0),end="\r")
        training  = [self.nets[l].train for l in range(10)]
        [f(imgs,self.y(lbls,i)) and \
                print("training [{:<10}] {}%".format("#"*(i+1),(i+1)*10),end="\r") \
                for i,f in enumerate(training)]
        print(" "*70,end="\r")
        

    def predict(self,img):
        predicting = [self.nets[l].predict for l in range(10)]
        m = map(lambda f: f(img), predicting)
        return map(lambda x: max(zip(x,range(10)))[1], zip(*m))

    def compute_error(self,imgs, lbls):
        m = map(lambda x,y: x==y, self.predict(imgs), lbls.A1)
        return 100-sum(m)/len(lbls)*100









