import random

import numpy as np

from losses.jFitnessFunction import jFitnessFunction

class jGreyWolfOptimizer:
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
                X[i, d] = lb + (ub - lb) * random.random()

        # Fitness
        fit = np.zeros(N)
        for i in range(N):
            fit[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)

        # Sort fitness
        idx = np.argsort(fit)
        # Update alpha, beta & delta
        Xalpha = X[idx[0], :]
        Xbeta = X[idx[1], :]
        Xdelta = X[idx[2], :]
        Falpha = fit[idx[0]]
        Fbeta = fit[idx[1]]
        Fdelta = fit[idx[2]]
        # Pre
        curve = np.zeros(self.max_Iter)
        curve[0] = Falpha
        t = 2
        # Iterations
        while t <= self.max_Iter:
            # Coefficient decreases linearly from 2 to 0
            a = 2 - t * (2 / self.max_Iter)
            for i in range(N):
                for d in range(dim):
                    # Parameter C (3.4)
                    C1 = 2 * random.random()
                    C2 = 2 * random.random()
                    C3 = 2 * random.random()
                    # Compute Dalpha, Dbeta & Ddelta (3.5)
                    Dalpha = abs(C1 * Xalpha[d] - X[i, d])
                    Dbeta = abs(C2 * Xbeta[d] - X[i, d])
                    Ddelta = abs(C3 * Xdelta[d] - X[i, d])
                    # Parameter A (3.3)
                    A1 = 2 * a * random.random() - a
                    A2 = 2 * a * random.random() - a
                    A3 = 2 * a * random.random() - a
                    # Compute X1, X2 & X3 (3.6)
                    X1 = Xalpha[d] - A1 * Dalpha
                    X2 = Xbeta[d] - A2 * Dbeta
                    X3 = Xdelta[d] - A3 * Ddelta
                    # Update wolf (3.7)
                    X[i, d] = (X1 + X2 + X3) / 3
                # Boundary
                XB = X[i, :]
                XB[XB > ub] = ub
                XB[XB < lb] = lb
                X[i, :] = XB

            # Fitness
            for i in range(N):
                # Fitness
                fit[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)
                # Update alpha, beta & delta
                if fit[i] < Falpha:
                    Falpha = fit[i]
                    Xalpha = X[i, :]
                if fit[i] < Fbeta and fit[i] > Falpha:
                    Fbeta = fit[i]
                    Xbeta = X[i, :]
                if fit[i] < Fdelta and fit[i] > Falpha and fit[i] > Fbeta:
                    Fdelta = fit[i]
                    Xdelta = X[i, :]

            curve[t - 1] = Falpha
            print('\nIteration %d Best (GWO)= %f' % (t, curve[t - 1]))
            t = t + 1

        # Select features based on selected index
        Pos = np.arange(dim)
        Sf = Pos[(Xalpha > self.thres) == 1]
        sFeat = x_train[:, Sf]
        # Store results
        GWO = {}
        GWO['sf'] = Sf
        GWO['c'] = curve

        return GWO