import numpy as np

from losses.jFitnessFunction import jFitnessFunction

class jFruitFlyOptimizationAlgorithm:
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
        Y = np.zeros((N, dim))
        for i in range(N):
            for d in range(dim):
                X[i, d] = lb + (ub - lb) * np.random.rand()
                Y[i, d] = lb + (ub - lb) * np.random.rand()

        # Compute solution
        S = np.zeros((N, dim))
        for i in range(N):
            for d in range(dim):
                # Distance between X and Y axis
                dist = np.sqrt(X[i, d] ** 2 + Y[i, d] ** 2)
                # Solution
                S[i, d] = 1 / dist
            # Boundary
            SB = S[i, :]
            SB[SB > ub] = ub
            SB[SB < lb] = lb
            S[i, :] = SB

        # Pre
        fit = np.zeros(N)
        fitG = np.inf
        curve = np.zeros(self.max_Iter)
        t = 1

        # Iterations
        while t <= self.max_Iter:
            # Fitness
            for i in range(N):
                # Fitness
                fit[i] = self.loss_func(x_train[:,S[i, :] > self.thres], x_test[:,S[i, :] > self.thres], y_train, y_test)
                # Update better solution
                if fit[i] < fitG:
                    fitG = fit[i]
                    Xgb = S[i, :]
                    # Update X & Y
                    Xb = X[i, :]
                    Yb = Y[i, :]

            for i in range(N):
                for d in range(dim):
                    # Random in [-1,1]
                    r1 = -1 + 2 * np.random.rand()
                    r2 = -1 + 2 * np.random.rand()
                    # Compute new X & Y
                    X[i, d] = Xb[d] + (ub - lb) * r1
                    Y[i, d] = Yb[d] + (ub - lb) * r2
                    # Distance between X and Y axis
                    dist = np.sqrt((X[i, d] ** 2) + (Y[i, d] ** 2))
                    # Solution
                    S[i, d] = 1 / dist
                # Boundary
                SB = S[i, :]
                SB[SB > ub] = ub
                SB[SB < lb] = lb
                S[i, :] = SB

            curve[t - 1] = fitG
            print('\nGeneration', t, 'Best (FOA)=', curve[t - 1])
            t = t + 1

        # Select features
        Pos = np.arange(0, dim)
        Sf = Pos[Xgb > self.thres]

        # Store results
        FOA = {}
        FOA['sf'] = Sf
        FOA['c'] = curve

        return FOA
