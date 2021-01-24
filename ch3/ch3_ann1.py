import torch
import numpy
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

n_dim=2
x_train, y_train= make_blobs(n_samples=80,n_features=n_dim, centers=[[1,1],[-1,-1],[1,-1],[-1,1]], shuffle=True, cluster_std=0.3)
x_test, y_test=make_blobs(n_samples=20,n_features=n_dim, centers=[[1,1],[-1,-1],[1,-1],[-1,1]], shuffle=True, cluster_std=0.3)

def label_map(y_, from_, to_):
    y=numpy.copy(y_)
    for f in from_:
        y[y_==f]=to_
    return y

y_train= label_map(y_train,[0,1],0)
y_train= label_map(y_train,[2,3],1)
y_test=label_map(y_test,[0,1],0)
y_test=label_map(y_test,[2,3],1)

def vis_data(x,y=None, c='r'):
    if y is None:
        y=[None]* len(x)
    for x_,y_ in zip(x,y):
        if y_ is None:
            plt.plot(x_[0], x_[1], '*', markerfacecolor='none', markeredgecolor=c)
        else:
            plt.plot(x_[0], x_[1], c+'o' if y_==0 else c+'+')

plt.figure()
vis_data(x_train,y_train, c='r')
plt.show()

x_train= torch.FloatTensor(x_train)
x_test=torch.FloatTensor(x_test)
y_train=torch.FloatTensor(y_train)
y_test=torch.FloatTensor(y_test)

class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self),__init()

        
