import numpy as np

from losses.jFitnessFunction import jFitnessFunction

class jFireflyAlgorithm:
    def __init__(self, N, max_Iter, loss_func, alpha=0.9, beta=0.1, thres=0.5, tau=1, rho=0.2, eta=1, beta0=1, gamma=1,theta=0.97):
        self.N = N
        self.max_Iter = max_Iter
        self.loss_func = loss_func
        self.alpha = alpha
        self.beta = beta
        self.thres = thres
        self.tau = tau
        self.rho = rho
        self.eta = eta
        self.beta0 = beta0
        self.gamma = gamma
        self.theta = theta

    def optimize(self, x_train, x_test, y_train, y_test):

        # Parameters
        lb = 0
        ub = 1
        beta0 = self.beta0  # light amplitude
        gamma = self.gamma  # absorbtion coefficient
        theta = self.theta  # control alpha
        N = self.N

        # Number of dimensions
        dim = np.size(x_train, 1)
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
            # Best solution
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i, :]

        # Pre
        curve = np.zeros(self.max_Iter)
        curve[0] = fitG
        t = 2

        # Generation
        while t <= self.max_Iter:
            # Alpha update
            alpha = self.alpha * theta
            # Rank firefly based on their light intensity
            idx = np.argsort(fit)
            X = X[idx, :]
            fit = fit[idx]

            for i in range(N):
                # The attractiveness parameter
                for j in range(N):
                    # Update moves if firefly j brighter than firefly i
                    if fit[i] > fit[j]:
                        # Compute Euclidean distance
                        r = np.sqrt(np.sum((X[i, :] - X[j, :]) ** 2))
                        # Beta (2)
                        beta = beta0 * np.exp(-gamma * r ** 2)

                        for d in range(dim):
                            # Update position (3)
                            eps = np.random.rand() - 0.5
                            X[i, d] = X[i, d] + beta * (X[j, d] - X[i, d]) + alpha * eps

                        # Boundary
                        XB = X[i, :]
                        XB[XB > ub] = ub
                        XB[XB < lb] = lb
                        X[i, :] = XB

                        # Fitness
                        fit[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)

                        # Update global best firefly
                        if fit[i] < fitG:
                            fitG = fit[i]
                            Xgb = X[i, :]

            curve[t - 1] = fitG
            print('\nGeneration', t, 'Best (FA)=', curve[t - 1])
            t = t + 1

        # Select features
        Pos = np.arange(0, dim)
        Sf = Pos[(Xgb > self.thres) == 1]

        # Store results
        FA = {'sf': Sf, 'c': curve}
        return FA
