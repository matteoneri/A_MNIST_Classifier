import numpy as np
from scipy import linalg
from scipy import sparse
from functools import reduce
from tensorflow.examples.tutorials.mnist import input_data

_mnst  = input_data.read_data_sets("../MNIST_data/", validation_size=10000)
imgs  = np.r_[_mnst.train.images,_mnst.test.images]
lbls   = np.r_[_mnst.train.labels,_mnst.test.labels]

imgs_val = np.r_[_mnst.validation.images]
lbls_val = _mnst.validation.labels

class neural_net:
    def __init__(self, m, n, sparse = False):
        self._m      = m
        self._n      = n
        self._R      = np.matrix(np.random.randn(m,n), dtype=np.float32)
        self._theta  = np.ones(m, dtype=np.float32)
        self._sparse = sparse

    @staticmethod
    def sigmoid(z):
        return 1/(1+np.exp(-z))

    def predict(self,x):
        if isinstance(x, sparse.csc_matrix):
            return self._theta*neural_net.sigmoid(self._R*x)*self._theta
        return np.dot(self._theta,neural_net.sigmoid(np.dot(self._R,x))).A1

    def train(self,x,y,gamma=0.1):
        gamma = np.float32(gamma)
        z = np.matrix(neural_net.sigmoid(np.dot(self._R,x)))
        self._theta = np.dot(np.linalg.inv(z*z.transpose()+gamma*np.eye(self._m, dtype=np.float32))*z,y)
        return "done"



class mnist_classifier:
    @staticmethod
    def y(i):
        return (lbls==i)*np.uint8(2)-np.ones(len(lbls),dtype=np.float32)
    x = np.transpose(imgs)
    
    def __init__(self,m):
        self.nets = [neural_net(m,28**2) for l in range(10)]
        training  = [self.nets[l].train for l in range(10)]
        [f(self.x,self.y(i)) and print(i) for i,f in enumerate(training)]
        
    def predict(self,img):
        predicting = [self.nets[l].predict for l in range(10)]
        m = [(f(img),i) for i,f in enumerate(predicting)]
        return max(m)[1]
    
    def predict2(self,img):
        predicting = [self.nets[l].predict for l in range(10)]
        m = map(lambda f: f(img), predicting)
        return map(lambda x: max(zip(x,range(10)))[1], zip(*m))
    
    def compute_error(self,imgs, lbls):
        m = map(lambda x,y: x==y, self.predict2(imgs), lbls)
        return 100-sum(m)/len(lbls)*100
            
            
            
            
            
            
            
            
            
