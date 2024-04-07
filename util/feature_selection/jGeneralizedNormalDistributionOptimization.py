import numpy as np

from losses.jFitnessFunction import jFitnessFunction

class jGeneralizedNormalDistributionOptimization:
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

        # Initial (26)
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
            # Best
            if fit[i] < fitG:
                fitG = fit[i]
                Xb = X[i, :]

        # Pre
        V = np.zeros((N, dim))

        curve = np.zeros(self.max_Iter)
        curve[0] = fitG
        t = 2
        # Iteration
        while t <= self.max_Iter:
            # Compute mean position (22)
            M = np.mean(X, axis=0)
            for i in range(N):
                alpha = np.random.rand()
                # [Local exploitation]
                if alpha > 0.5:
                    # Random numbers
                    a = np.random.rand()
                    b = np.random.rand()
                    for d in range(dim):
                        # Compute mean (19)
                        mu = (1 / 3) * (X[i, d] + Xb[d] + M[d])
                        # Compute standard deviation (20)
                        delta = np.sqrt((1 / 3) * ((X[i, d] - mu) ** 2 +
                                                   (Xb[d] - mu) ** 2 +
                                                   (M[d] - mu) ** 2))
                        # Compute eta (21)
                        lambda1 = np.random.rand()
                        lambda2 = np.random.rand()
                        if a <= b:
                            eta = np.sqrt(-1 * np.log(lambda1)) * np.cos(2 * np.pi * lambda2)
                        else:
                            eta = np.sqrt(-1 * np.log(lambda1)) * np.cos(2 * np.pi * lambda2 + np.pi)
                        # Generate normal ditribution (18)
                        V[i, d] = mu + delta * eta
                # [Global Exploitation]
                else:
                    # Random three vectors but not i
                    RN = np.random.permutation(N)
                    RN = RN[RN != i]
                    p1 = RN[0]
                    p2 = RN[1]
                    p3 = RN[2]
                    # Random beta
                    beta = np.random.rand()
                    # Normal random number: zero mean & unit variance
                    lambda3 = np.random.randn()
                    lambda4 = np.random.randn()
                    # Get v1 (24)
                    if fit[i] < fit[p1]:
                        v1 = X[i, :] - X[p1, :]
                    else:
                        v1 = X[p1, :] - X[i, :]
                    # Get v2 (25)
                    if fit[p2] < fit[p3]:
                        v2 = X[p2, :] - X[p3, :]
                    else:
                        v2 = X[p3, :] - X[p2, :]
                    # Generate new position (23)
                    for d in range(dim):
                        V[i, d] = X[i, d] + beta * (abs(lambda3) * v1[d]) + \
                                  (1 - beta) * (abs(lambda4) * v2[d])
                # Boundary
                XB = V[i, :]
                XB[XB > ub] = ub
                XB[XB < lb] = lb
                V[i, :] = XB
            # Fitness
            for i in range(N):
                fitV = self.loss_func(x_train[:, V[i, :] > self.thres], x_test[:, V[i, :] > self.thres],
                                        y_train, y_test)
                # Greedy selection (27)
                if fitV < fit[i]:
                    fit[i] = fitV
                    X[i, :] = V[i, :]
                # Best
                if fit[i] < fitG:
                    fitG = fit[i]
                    Xb = X[i, :]
            # Save
            curve[t - 1] = fitG
            print('\nIteration', t, 'Best (GNDO)=', curve[t - 1])
            t = t + 1

        # Select features based on selected index
        Pos = np.arange(dim)
        Sf = Pos[Xb > self.thres]

        # Store results
        GNDO = {'sf': Sf, 'c': curve}

        return GNDO