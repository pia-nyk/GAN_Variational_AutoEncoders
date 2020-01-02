import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal as mvn

def softplus(x):
    #ln(1 + exp(x))
    return np.log1p(np.exp(x))

#input layer of dimension 4, hidden layer of dimension 3, output layer of dimension 2
W1 = np.random.randn(4,3)
W2 = np.random.randn(3,2*2)

#2*2 because 2 o/p nodes for mean and 2 for std.dev

def forward_propagation(x,W1,W2):
    hidden = np.tanh(x.dot(W1))
    output = hidden.dot(W2)
    mean = output[:2]
    std_dev = softplus(output[2:])
    return mean, std_dev

#random input
x = np.random.randn(4)

mean, std_dev = forward_propagation(x,W1,W2)
print("mean: ", mean)
print("std dev: ", std_dev)

#draw the samples
samples = mvn.rvs(mean=mean, cov=std_dev**2, size=10000)

#plot the samples
plt.scatter(samples[:,0], samples[:,1], alpha=0.5)
plt.show()
