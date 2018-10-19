import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('ex2data1.txt', delimiter=',', names=('Score_1','Score_2','y'))

X = df.ix[:,0:2]
y = df.ix[:,2]
m=len(y)
X.insert(loc=0, column='A', value=np.ones([len(y),1]))
y = y[:,np.newaxis]
theta = np.zeros([3,1])
alpha = 0.03
iters = 1000

def sigmoid(x, theta):
    return 1/(1+np.exp(-(np.dot(x, theta))))

def cost(X, y, theta):
    J = (-1 / m) * np.sum(np.multiply(y, np.log(sigmoid(X, theta)))
                          + np.multiply((1 - y), np.log(1 - sigmoid(X, theta))))
    return J

def gradient(X, y, theta):
    for i in range(iters):
        sub = (alpha/m)*(np.sum(np.multiply((sigmoid(X,theta)-y),X)))
        theta = np.subtract(theta, np.array(sub).transpose())
    return theta

#theta = gradient(X,y,theta)
#print(cost(X, y, theta))
print(gradient(X,y,theta))