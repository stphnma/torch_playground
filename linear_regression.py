from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np


class LinearRegression():

    def __init__(self):
        pass

    def fit(self, X, y):

        N,K = X.shape

        inputs = Variable(X)
        actual = Variable(y)

        criterion = nn.MSELoss()

        linear = torch.nn.Linear(K, 1, bias=True)
        optimizer = optim.Adam(linear.parameters())

        for epoch in range(10000):

            outputs = linear(inputs)
            loss = criterion(outputs, actual)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(linear.weight)


    def predict(self, X):
        pass



if __name__ == "__main__":

    N = 500
    K = 5

    mean = np.random.normal(5, .2, size = K)

    X = np.random.multivariate_normal(mean = mean, cov = np.identity(K), size = (N))
    B = np.random.uniform( size = K)
    y = X.dot(B)# + np.random.normal(0,1, size = N)

    X_torch = torch.from_numpy(X).float()
    y_torch = torch.from_numpy(y).float()

    Linear = LinearRegression()
    Linear.fit(X_torch, y_torch)

    print("actual Beta", B)
