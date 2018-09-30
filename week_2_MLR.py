import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('ex1data2.txt', delimiter=',', names=('Size','Bedrooms','Price'))
w = df.ix[:,0:2]
y = df.ix[:,2]
y = y[:,np.newaxis]
alpha = 0.03
iters = 1000
theta = np.zeros([3,1])

Xm = np.mean(w)
Xs = np.std(w)
X = (w - Xm)/Xs

X.insert(loc=0, column='A', value=np.ones([len(y),1]))
m=len(y)

def gradientDescent(X , y, theta, alpha, iters):
    Xt = np.transpose(X)
    for i in range(0,iters):
        h = np.dot(X, theta) - y
        h = np.dot(Xt, h)
        theta = theta - h*(alpha/m)
    return theta


def computeCost(X, y, theta):
    h = np.dot(X,theta) - y
    diff = np.power(h,2)
    sum = np.sum(diff)
    J = sum/(2*m)
    return J

J = computeCost(X, y, theta)
print(J)

q = gradientDescent(X, y, theta, alpha, iters)
J = computeCost(X, y, q)

print(J)