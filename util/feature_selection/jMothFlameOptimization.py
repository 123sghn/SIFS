import numpy as np

from losses.jFitnessFunction import jFitnessFunction

class jMothFlameOptimization:
    def __init__(self, N, max_Iter, loss_func, alpha=0.9, beta=0.1, thres=0.5, tau=1, rho=0.2, eta=1,b=1):
        self.N = N
        self.max_Iter = max_Iter
        self.loss_func = loss_func
        self.alpha = alpha
        self.beta = beta
        self.thres = thres
        self.tau = tau
        self.rho = rho
        self.eta = eta
        self.b=b

    def optimize(self, x_train, x_test, y_train, y_test):
        # Parameters
        lb = 0
        ub = 1
        b = self.b     # constant
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
        fitG = float('inf')

        curve = np.zeros(self.max_Iter)
        t = 1
        while t <= self.max_Iter:
            for i in range(N):
                # Fitness
                fit[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)
                # Global best
                if fit[i] < fitG:
                    fitG = fit[i]
                    Xgb = X[i, :]

            if t == 1:
                # Best flame
                idx = np.argsort(fit)
                fitF = fit[idx]
                flame = X[idx, :]
            else:
                # Sort population
                XX = np.concatenate((flame, X), axis=0)
                FF = np.concatenate((fitF, fit), axis=0)
                idx = np.argsort(FF)
                flame = XX[idx[:N], :]
                fitF = FF[:N]

            # Flame update (3.14)
            flame_no = round(N - t * ((N - 1) / self.max_Iter))
            # Convergence constant, decreases linearly from -1 to -2
            r = -1 + t * (-1 / self.max_Iter)

            for i in range(N):
                # Normal position update
                if i <= flame_no:
                    for d in range(dim):
                        # Parameter T0, from r to 1
                        T = (r - 1) * np.random.rand() + 1
                        # Distance between flame & moth (3.13)
                        dist = np.abs(flame[i, d] - X[i, d])
                        # Moth update (3.12)
                        X[i, d] = dist * np.exp(b * T) * np.cos(2 * np.pi * T) + flame[i, d]
                # Position update respect to best flames
                else:
                    for d in range(dim):
                        # Parameter T, from r to 1
                        T = (r - 1) * np.random.rand() + 1
                        # Distance between flame & moth (3.13)
                        dist = np.abs(flame[i, d] - X[i, d])
                        # Moth update (3.12)
                        X[i, d] = dist * np.exp(b * T) * np.cos(2 * np.pi * T) + flame[flame_no, d]

                # Boundary
                XB = X[i, :]
                XB[XB > ub] = ub
                XB[XB < lb] = lb
                X[i, :] = XB

            curve[t-1] = fitG
            print('\nIteration {} Best (MPA)= {}'.format(t, curve[t-1]))
            t = t + 1

        # Select features
        Pos = np.arange(dim)
        Sf = Pos[(Xgb > self.thres) == 1]

        # Store results
        MFO = {
            'sf': Sf,
            'c': curve
        }

        return MFO