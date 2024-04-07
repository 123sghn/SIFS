import numpy as np

from losses.jFitnessFunction import jFitnessFunction

class jWeightedSuperpositionAttraction:
    def __init__(self, N, max_Iter, loss_func, alpha=0.9, beta=0.1, thres=0.5, tau=1, rho=0.2, eta=1, sl=0.035, phi=0.001, lambda_val=0.75):
        self.N = N
        self.max_Iter = max_Iter
        self.loss_func = loss_func
        self.alpha = alpha
        self.beta = beta
        self.thres = thres
        self.tau = tau
        self.rho = rho
        self.eta = eta
        self.sl =sl
        self.phi=phi
        self.lambda_val=lambda_val

    def optimize(self, x_train, x_test, y_train, y_test):

        # 参数
        lb = 0
        ub = 1
        sl = self.sl  # step length
        phi = self.phi  # constant
        lambda_val = self.lambda_val  # constant
        N = self.N

        # Objective function
        def fun(feat, label, X, opts):
            return jFitnessFunction(feat, label, (X > self.thres), opts)

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
            # Best update
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i, :]

        # Pre
        curve = np.zeros(self.max_Iter)
        curve[0] = fitG
        t = 2

        # Iterations
        while t <= self.max_Iter:
            # Rank solution based on fitness
            idx = np.argsort(fit)
            fit = fit[idx]
            X = X[idx, :]

            # {1} Target point determination: Figure 2
            w = np.zeros(N)
            Xtar = np.zeros(dim)
            for i in range(1,N+1):
                # Assign weight based on rank
                w[i-1] = i ** (-1 * self.tau)
                # Create target
                for d in range(dim):
                    Xtar[d] = Xtar[d] + X[i-1, d] * w[i-1]

            # Boundary
            Xtar[Xtar > ub] = ub
            Xtar[Xtar < lb] = lb

            # Fitness
            fitT = self.loss_func(x_train[:, Xtar > self.thres], x_test[:, Xtar > self.thres],
                                        y_train, y_test)

            # Best update
            if fitT < fitG:
                fitG = fitT
                Xgb = Xtar

            # {2} Compute search direction: Figure 4
            gap = np.zeros((N, dim))
            direct = np.zeros((N, dim))
            for i in range(N):
                if fit[i] >= fitT:
                    for d in range(dim):
                        # Compute gap
                        gap[i, d] = Xtar[d] - X[i, d]
                        # Compute direction
                        direct[i, d] = np.sign(gap[i, d])
                elif fit[i] < fitT:
                    if np.random.rand() < np.exp(fit[i] - fitT):
                        for d in range(dim):
                            # Compute gap
                            gap[i, d] = Xtar[d] - X[i, d]
                            # Compute direction
                            direct[i, d] = np.sign(gap[i, d])
                    else:
                        for d in range(dim):
                            # Compute direction
                            direct[i, d] = np.sign(-1 + (1 + 1) * np.random.rand())

            # Compute step sizing function (2)
            if np.random.rand() <= lambda_val:
                sl = sl - np.exp(t / (t - 1)) * phi * sl
            else:
                sl = sl + np.exp(t / (t - 1)) * phi * sl

            # {3} Neighbor generation: Figure 7
            for i in range(N):
                for d in range(dim):
                    # Update (1)
                    X[i, d] = X[i, d] + sl * direct[i, d] * np.abs(X[i, d])
                # Boundary
                X[X > ub] = ub
                X[X < lb] = lb

            # Fitness
            for i in range(N):
                # Fitness
                fit[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)
                # Best update
                if fit[i] < fitG:
                    fitG = fit[i]
                    Xgb = X[i, :]

            curve[t - 1] = fitG
            print('\nIteration', t, 'Best (WSA)=', curve[t - 1])
            t = t + 1

        # Select features
        Pos = np.arange(dim)
        Sf = Pos[(Xgb > self.thres) == 1]

        # Store results
        WSA = {}
        WSA['sf'] = Sf
        WSA['c'] = curve

        return WSA