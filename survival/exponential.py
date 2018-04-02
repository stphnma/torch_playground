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


        # import ipdb; ipdb.set_trace()
        optimizer = optim.Adam([Betas, alpha])
        for epoch in range(num_epochs):
            log_survival = duration *  -1 * torch.exp( alpha + torch.mm(inputs, Betas) )
            log_pdf = torch.log(alpha) + torch.mm(inputs, Betas) * actual
            NLL = -1 * torch.sum(log_survival + log_pdf)
            optimizer.zero_grad()
            NLL.backward(retain_graph = True)
            optimizer.step()
            #
            print(NLL.data[0])
        print(Betas)


def _test_exponential():

    lambda_0 = 10
    B1, B2 = 1, 2
    max_T = 50
    N = 10000

    X = []
    for i in range(N):
        X.append([1,0])
        X.append([0,1])
        X.append([0,0])
        X.append([1,1])


    T = [max_T] * len(X)
    C = [None] * len(X)

    for i, x in enumerate(X):
        lambda_ = lambda_0 * np.exp(np.matmul(x, [B1,B2]))
        t = np.random.exponential(1/lambda_)
        if t < max_T:
            T[i] = t
            C[i] = 1
        else:
            C[i] = 0

    df = pd.DataFrame(X)
    df.columns = ['X1','X2']

    X_torch = torch.from_numpy(df.values).float()
    y_torch = torch.from_numpy(pd.DataFrame(C).values).float()
    T_torch = torch.from_numpy(T.values).float()


    df['duration'] = T

    y = pd.Series(C)

    print "Simulated Values: %d" %len(X)

    S = ExponentialRegressionClassifier()
    S.fit(df, y)


if __name__ == "__main__":

    from lifelines.datasets import load_rossi
    from lifelines import CoxPHFitter
    rossi_dataset = load_rossi()

    T = rossi_dataset[['week']]
    y = rossi_dataset[['arrest']]
    X = rossi_dataset[[x for x in rossi_dataset.columns if x not in ['week','arrest']]]

    X_torch = torch.from_numpy(X.values).float()
    y_torch = torch.from_numpy(y.values).float()
    T_torch = torch.from_numpy(T.values).float()

    cph = CoxPHFitter()
    cph.fit(rossi_dataset, duration_col='week', event_col='arrest')
    cph.print_summary()  # access the results using cph.summary

    Exp = Exponential()
    Exp.fit(X_torch, T_torch, y_torch)
