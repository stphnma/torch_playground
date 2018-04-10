import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

class Exponential():

    def __init__(self):
        self.weights = None


    def fit(self, X, T, y):

        num_epochs = 20000
        N,K = X.shape
        inputs = Variable(X)
        duration = Variable(T)
        actual = Variable(y)

        Betas = Variable(torch.rand(K,1), requires_grad = True)
        alpha = Variable(torch.rand(1), requires_grad = True)
        optimizer = optim.Adam([Betas, alpha])

        _betas = []

        for epoch in range(num_epochs):
            log_survival = duration *  -1 * torch.exp( alpha + torch.mm(inputs, Betas) )
            log_pdf = torch.log(alpha) + torch.mm(inputs, Betas) * actual
            NLL = -1 * torch.sum(log_survival + log_pdf)
            optimizer.zero_grad()
            NLL.backward(retain_graph = True)
            optimizer.step()
            #
            print(NLL.data[0], Betas.data[0][0], alpha.data[0])

            _betas.append(Betas.data[0][0])


        import ipdb; ipdb.set_trace()


def simulate(lambda_0, Betas, max_T):

    pass

def test(lambda_0 = 10, B1 = 1, max_T = 50):

    X = []
    for i in range(10000):
        X.append([1])
        X.append([0])

    T = [max_T] * len(X)
    C = [0] * len(X)

    for i, x in enumerate(X):
        lambda_ = lambda_0 * np.exp(np.matmul(x, [B1]))
        t = np.random.exponential(1/lambda_)
        if t < max_T:
            T[i] = t
            C[i] = 1

    N = len(X)
    X = np.array(X)
    y = np.array(C).reshape(N, 1)
    T = np.array(T).reshape(N, 1)

    X_torch = torch.from_numpy(X).float()
    y_torch = torch.from_numpy(y).float()
    T_torch = torch.from_numpy(T).float()

    Exp = Exponential()
    Exp.fit(X_torch, T_torch, y_torch)


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

    test()
