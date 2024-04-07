import numpy as np

from losses.jFitnessFunction import jFitnessFunction

class jWhaleOptimizationAlgorithm:
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
        b = self.b  # constant
        N = self.N

        # Number of dimensions
        dim = x_train.shape[1]
        # Initial
        X = np.zeros((N, dim))
        for i in range(N):
            for d in range(dim):
                X[i, d] = lb + (ub - lb) * np.random.rand()

        # Fitness
        fit = np.zeros(N)
        fitG = np.inf
        for i in range(N):
            fit[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)
            # Global best
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i, :].copy()

        # Pre
        curve = np.zeros(self.max_Iter)
        curve[0] = fitG
        t = 2
        while t <= self.max_Iter:
            # Define a, linearly decreases from 2 to 0
            a = 2 - t * (2 / self.max_Iter)
            for i in range(N):
                # Parameter A (2.3)
                A = 2 * a * np.random.rand() - a
                # Parameter C (2.4)
                C = 2 * np.random.rand()
                # Parameter p, random number in [0,1]
                p = np.random.rand()
                # Parameter l, random number in [-1,1]
                l = -1 + 2 * np.random.rand()
                # Whale position update (2.6)
                if p < 0.5:
                    # {1} Encircling prey
                    if abs(A) < 1:
                        for d in range(dim):
                            # Compute D (2.1)
                            Dx = abs(C * Xgb[d] - X[i, d])
                            # Position update (2.2)
                            X[i, d] = Xgb[d] - A * Dx
                    # {2} Search for prey
                    elif abs(A) >= 1:
                        for d in range(dim):
                            # Select a random whale
                            k = np.random.randint(0, N)
                            # Compute D (2.7)
                            Dx = abs(C * X[k, d] - X[i, d])
                            # Position update (2.8)
                            X[i, d] = X[k, d] - A * Dx
                # {3} Bubble-net attacking
                elif p >= 0.5:
                    for d in range(dim):
                        # Distance of whale to prey
                        dist = abs(Xgb[d] - X[i, d])
                        # Position update (2.5)
                        X[i, d] = dist * np.exp(b * l) * np.cos(2 * np.pi * l) + Xgb[d]

                # Boundary
                X[i, :] = np.clip(X[i, :], lb, ub)

            # Fitness
            for i in range(N):
                # Fitness
                fit[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)
                # Global best
                if fit[i] < fitG:
                    fitG = fit[i]
                    Xgb = X[i, :].copy()

            curve[t - 1] = fitG
            print('\nIteration %d Best (WOA)= %f' % (t, curve[t - 1]))
            t += 1

        # Select features
        Pos = np.arange(dim)
        Sf = Pos[(Xgb > self.thres) == 1]

        # Store results
        WOA = {}
        WOA['sf'] = Sf.tolist()
        WOA['c'] = curve.tolist()

        return WOA