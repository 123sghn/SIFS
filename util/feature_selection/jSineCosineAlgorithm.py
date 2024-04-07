import numpy as np

from losses.jFitnessFunction import jFitnessFunction

class jSineCosineAlgorithm:
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

        fitD = np.inf
        fit = np.zeros(N)

        curve = np.zeros(self.max_Iter)
        t = 1
        # Iterations
        while t <= self.max_Iter:
            # Destination point
            for i in range(N):
                # Fitness
                fit[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)
                # Destination update
                if fit[i] < fitD:
                    fitD = fit[i]
                    Xdb = X[i, :]

            # Parameter r1, decreases linearly from alpha to 0 (3.4)
            r1 = self.alpha - t * (self.alpha / self.max_Iter)
            for i in range(N):
                for d in range(dim):
                    # Random parameter r2 & r3 & r4
                    r2 = (2 * np.pi) * np.random.rand()
                    r3 = 2 * np.random.rand()
                    r4 = np.random.rand()
                    # Position update (3.3)
                    if r4 < 0.5:
                        # Sine update (3.1)
                        X[i, d] = X[i, d] + r1 * np.sin(r2) * np.abs(r3 * Xdb[d] - X[i, d])
                    else:
                        # Cosine update (3.2)
                        X[i, d] = X[i, d] + r1 * np.cos(r2) * np.abs(r3 * Xdb[d] - X[i, d])

                    # Boundary
                    X[i, :] = np.clip(X[i, :], lb, ub)

            curve[t - 1] = fitD
            print('\nIteration', t, 'Best (SCA)=', curve[t - 1])
            t = t + 1

        # Select features
        Pos = np.arange(dim)
        Sf = Pos[(Xdb > self.thres) == 1]

        # Store results
        SCA = {}
        SCA['sf'] = Sf
        SCA['c'] = curve
        return SCA
