import numpy as np

from losses.jFitnessFunction import jFitnessFunction

class jJayaAlgorithm:
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
        # Fitness
        fit = np.zeros(N)
        fitG = np.inf
        for i in range(N):
            fit[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)
            # Best
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i, :]
        # Pre
        Xnew = np.zeros((N, dim))
        curve = np.zeros(self.max_Iter)
        curve[0] = fitG
        t = 2
        # Iteration
        while t <= self.max_Iter:
            # Identify best & worst in population
            idxB = np.argmin(fit)
            Xbest = X[idxB, :]
            idxW = np.argmax(fit)
            Xworst = X[idxW, :]
            # Start
            for i in range(N):
                for d in range(dim):
                    # Random numbers
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    # Update (1)
                    Xnew[i, d] = X[i, d] + r1 * (Xbest[d] - np.abs(X[i, d])) - r2 * (Xworst[d] - np.abs(X[i, d]))
                # Boundary
                Xnew[i, :] = np.clip(Xnew[i, :], lb, ub)
            # Fitness
            for i in range(N):
                Fnew = self.loss_func(x_train[:, Xnew[i, :] > self.thres], x_test[:, Xnew[i, :] > self.thres],
                                        y_train, y_test)
                # Greedy selection
                if Fnew < fit[i]:
                    fit[i] = Fnew
                    X[i, :] = Xnew[i, :]
                # Best
                if fit[i] < fitG:
                    fitG = fit[i]
                    Xgb = X[i, :]
            # Save
            curve[t - 1] = fitG
            print('\nIteration', t, 'Best (JA)=', curve[t - 1])
            t = t + 1
        # Select features based on selected index
        Pos = np.arange(dim)
        Sf = Pos[(Xgb > self.thres) == 1]
        sFeat = x_train[:, Sf]
        JA = {}
        JA['sf'] = Sf
        JA['c'] = curve
        return JA