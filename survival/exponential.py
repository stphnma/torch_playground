import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

class Exponential():

    def __init__(self):
        self.weights = None


    def fit(self, X, T, y, verbose = False):

        num_epochs = 50000
        N,K = X.shape
        inputs = Variable(X)
        duration = Variable(T)
        actual = Variable(y)

        # Betas = Variable(torch.rand(K,1), requires_grad = True)
        alpha = torch.rand(1, requires_grad = True)
        # Variable(torch.rand(1), requires_grad = True)
        # optimizer = optim.Adam([Betas, alpha])
        optimizer = optim.Adam([alpha])

        _betas = []

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            loglik =  actual * torch.log(alpha) -  alpha * duration
            NLL = -1 * torch.sum(loglik)
            optimizer.zero_grad()
            NLL.backward(retain_graph = True)
            optimizer.step()

            if verbose:
                if epoch % 1000 == 0:
                    print(alpha)

        self.alpha = alpha.item()



def test_exponential(lambda_0 = 10, B1 = 1, max_T = 50):

    size = 10000

    T = np.random.exponential(1/lambda_0, size = size)
    event = lambda x: 1 if x <= max_T else 0
    C = [event(t) for t in T]
    T = [min(t, max_T) for t in T]
    X = [0] * size

    N = len(X)
    X = np.array(X).reshape(N, 1)
    y = np.array(C).reshape(N, 1)
    T = np.array(T).reshape(N, 1)

    X_torch = torch.from_numpy(X).float()
    y_torch = torch.from_numpy(y).float()
    T_torch = torch.from_numpy(T).float()

    Exp = Exponential()
    Exp.fit(X_torch, T_torch, y_torch, verbose = True)

    alpha = Exp.alpha
    assert abs(alpha - lambda_0) < 0.05

if __name__ == "__main__":

    # from lifelines.datasets import load_rossi
    # from lifelines import CoxPHFitter
    # rossi_dataset = load_rossi()
    #
    # T = rossi_dataset[['week']]
    # y = rossi_dataset[['arrest']]
    # X = rossi_dataset[[x for x in rossi_dataset.columns if x not in ['week','arrest']]]
    #
    # X_torch = torch.from_numpy(X.values).float()
    # y_torch = torch.from_numpy(y.values).float()
    # T_torch = torch.from_numpy(T.values).float()
    #
    # cph = CoxPHFitter()
    # cph.fit(rossi_dataset, duration_col='week', event_col='arrest')
    # cph.print_summary()  # access the results using cph.summary

    # Exp = Exponential()
    # Exp.fit(X_torch, T_torch, y_torch)

    test_exponential()
