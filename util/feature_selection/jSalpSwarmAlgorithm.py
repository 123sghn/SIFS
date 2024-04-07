import numpy as np

from losses.jFitnessFunction import jFitnessFunction

class jSalpSwarmAlgorithm:
    def __init__(self, N, max_Iter, loss_func, alpha=0.9, beta=0.1, thres=0.5, tau=1, rho=0.2, eta=1):
        self.N = N
        self.max_Iter = max_Iter
        self.loss_func = loss_func
        self.alpha = alpha
        self.beta = beta
        self.thres = thres
        self.tau = tau
        self.rho = rho
        self.eta = eta

    def optimize(self, x_train, x_test, y_train, y_test):

        # Parameters
        lb = 0
        ub = 1
        N = self.N

        # Number of dimensions
        dim = x_train.shape[1]
        # Initial
        X = np.zeros((N, dim))
        for i in range(N):
            for d in range(dim):
                X[i, d] = lb + (ub - lb) * np.random.rand()

        # Pre
        fit = np.zeros(N)
        fitF = np.inf

        curve = np.zeros(self.max_Iter)
        t = 1
        # Iteration
        while t <= self.max_Iter:
            for i in range(N):
                # Fitness
                fit[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)
                # Best food update
                if fit[i] < fitF:
                    Xf = X[i, :]
                    fitF = fit[i]
            # Compute coefficient, c1 (3.2)
            c1 = 2 * np.exp(- (4 * t / self.max_Iter) ** 2)
            for i in range(N):
                # Leader update
                if i == 0:
                    for d in range(dim):
                        # Coefficient c2 & c3 [0~1]
                        c2 = np.random.rand()
                        c3 = np.random.rand()
                        # Leader update (3.1)
                        if c3 >= 0.5:
                            X[i, d] = Xf[d] + c1 * ((ub - lb) * c2 + lb)
                        else:
                            X[i, d] = Xf[d] - c1 * ((ub - lb) * c2 + lb)
                # Salp update
                elif i >= 1:
                    for d in range(dim):
                        # Salp update by following front salp (3.4)
                        X[i, d] = (X[i, d] + X[i - 1, d]) / 2
                # Boundary
                X[i, :] = np.clip(X[i, :], lb, ub)

            curve[t - 1] = fitF
            print('\nIteration %d Best (SSA) = %f' % (t, curve[t - 1]))
            t = t + 1

        # Select features
        Pos = np.arange(1, dim + 1)
        Sf = Pos[(Xf > self.thres)]

        # Store results
        SSA = {}
        SSA['sf'] = Sf
        SSA['c'] = curve

        return SSA