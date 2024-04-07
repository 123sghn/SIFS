import numpy as np
from losses.jFitnessFunction import jFitnessFunction


class jEmperorPenguinOptimizer:
    def __init__(
        self,
        N,
        max_Iter,
        loss_func,
        alpha=0.9,
        beta=0.1,
        thres=0.5,
        tau=1,
        rho=0.2,
        eta=1,
        M=2,
        f=3,
        l=2,
    ):
        self.N = N
        self.max_Iter = max_Iter
        self.loss_func = loss_func
        self.alpha = alpha
        self.beta = beta
        self.thres = thres
        self.tau = tau
        self.rho = rho
        self.eta = eta
        self.M = M
        self.f = f
        self.l = l

    def optimize(self, x_train, x_test, y_train, y_test):
        # Parameters
        lb = 0
        ub = 1
        M = self.M  # movement parameter
        f = self.f  # control parameter
        l = self.l  # control parameter
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
        fitG = np.inf

        curve = np.zeros(self.max_Iter)
        t = 1
        # Iterations
        while t <= self.max_Iter:
            for i in range(N):
                # Fitness
                fit[i] = self.loss_func(
                    x_train[:, X[i, :] > self.thres],
                    x_test[:, X[i, :] > self.thres],
                    y_train,
                    y_test,
                )
                # Best solution
                if fit[i] < fitG:
                    fitG = fit[i]
                    Xgb = X[i, :].copy()
            # Generate radius in [0,1]
            R = np.random.rand()
            # Time (7)
            if R > 1:
                T0 = 0
            else:
                T0 = 1
            # Temperature profile (7)
            T = T0 - (self.max_Iter / (t - 1 - self.max_Iter))
            for i in range(N):
                for d in range(dim):
                    # Pgrid (10)
                    P_grid = abs(Xgb[d] - X[i, d])
                    # Vector A (9)
                    A = (M * (T + P_grid) * np.random.rand()) - T
                    # Vector C (11)
                    C = np.random.rand()
                    # Compute function S (12)
                    S = np.sqrt(f * np.exp(t / l) - np.exp(-t)) ** 2
                    # Distance (8)
                    Dep = abs(S * Xgb[d] - C * X[i, d])
                    # Position update (13)
                    X[i, d] = Xgb[d] - A * Dep
                # Boundary
                XB = X[i, :].copy()
                XB[XB > ub] = ub
                XB[XB < lb] = lb
                X[i, :] = XB
            curve[t - 1] = fitG
            print("\nIteration %d Best (EPO)= %f" % (t, curve[t - 1]))
            t = t + 1
        # Select features
        Pos = np.arange(dim)
        Sf = Pos[(Xgb > self.thres) == 1]
        # Store results
        EPO = {}
        EPO["sf"] = Sf
        EPO["c"] = curve

        return EPO
