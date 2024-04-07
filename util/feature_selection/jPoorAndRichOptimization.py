import numpy as np

from losses.jFitnessFunction import jFitnessFunction

class jPoorAndRichOptimization:
    def __init__(self, N, max_Iter, loss_func, alpha=0.9, beta=0.1, thres=0.5, tau=1, rho=0.2, eta=1, Pmut=0.06):
        self.N = N
        self.max_Iter = max_Iter
        self.loss_func = loss_func
        self.alpha = alpha
        self.beta = beta
        self.thres = thres
        self.tau = tau
        self.rho = rho
        self.eta = eta
        self.Pmut = Pmut

    def optimize(self, x_train, x_test, y_train, y_test):

        # Parameters
        lb = 0
        ub = 1
        Pmut = self.Pmut  # mutation probability

        N = self.N

        # Double population size: Main = Poor + Rich (1)
        N = N + N
        # Objective function
        fun = jFitnessFunction
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
        # Sort poor & rich (2)
        idx = np.argsort(fit)
        fit = fit[idx]
        X = X[idx, :]
        # Pre
        XRnew = np.zeros((N // 2, dim))
        XPnew = np.zeros((N // 2, dim))
        fitRnew = np.zeros(N // 2)
        fitPnew = np.zeros(N // 2)

        curve = np.zeros(self.max_Iter)
        curve[0] = fitG
        t = 1
        # Iteration
        while t < self.max_Iter:
            # Divide poor & rich
            XR = X[0:N // 2, :]
            fitR = fit[0:N // 2]
            XP = X[N // 2:N, :]
            fitP = fit[N // 2:N]
            # Select best rich individual
            idxR = np.argmin(fitR)
            XR_best = XR[idxR, :]
            # Select best poor individual
            idxP = np.argmin(fitP)
            XP_best = XP[idxP, :]
            # Compute mean of rich
            XR_mean = np.mean(XR, axis=0)
            # Compute worst of rich
            idxW = np.argmax(fitR)
            XR_worst = XR[idxW, :]
            # [Rich population]
            for i in range(N // 2):
                for d in range(dim):
                    # Generate new rich (3)
                    XRnew[i, d] = XR[i, d] + np.random.rand() * (XR[i, d] - XP_best[d])
                    # Mutation (6)
                    if np.random.rand() < Pmut:
                        # Normal random number with mean = 0 & sd = 1
                        G = 0 + 1 * np.random.randn()
                        # Mutation
                        XRnew[i, d] = XRnew[i, d] + G
                # Boundary
                XB = XRnew[i, :].copy()
                XB[XB > ub] = ub
                XB[XB < lb] = lb
                XRnew[i, :] = XB
                # Fitness of new rich
                fitRnew[i] = self.loss_func(x_train[:, XRnew[i, :] > self.thres], x_test[:, XRnew[i, :] > self.thres],
                                        y_train, y_test)
            # [Poor population]
            for i in range(N // 2):
                for d in range(dim):
                    # Calculate pattern (5)
                    pattern = (XR_best[d] + XR_mean[d] + XR_worst[d]) / 3
                    # Generate new poor (4)
                    XPnew[i, d] = XP[i, d] + (np.random.rand() * pattern - XP[i, d])
                    # Mutation (7)
                    if np.random.rand() < Pmut:
                        # Normal random number with mean = 0 & sd = 1
                        G = 0 + 1 * np.random.randn()
                        # Mutation
                        XPnew[i, d] = XPnew[i, d] + G
                # Boundary
                XB = XPnew[i, :].copy()
                XB[XB > ub] = ub
                XB[XB < lb] = lb
                XPnew[i, :] = XB
                # Fitness of new poor
                fitPnew[i] = self.loss_func(x_train[:, XPnew[i, :] > self.thres], x_test[:, XPnew[i, :] > self.thres],
                                        y_train, y_test)
            # Merge all four groups
            X = np.concatenate((XR, XP, XRnew, XPnew))
            fit = np.concatenate((fitR, fitP, fitRnew, fitPnew))
            # Select the best N individual
            idx = np.argsort(fit)
            fit = fit[idx][:N]
            X = X[idx[:N], :]
            # Best update
            if fit[0] < fitG:
                fitG = fit[0]
                Xgb = X[0, :]
            curve[t] = fitG
            print('\nIteration %d Best (PRO)= %f' % (t, curve[t]))
            t = t + 1
        # Select features based on selected index
        Pos = np.arange(dim)
        Sf = Pos[(Xgb > self.thres) == 1]
        # Store results
        PRO = {}
        PRO['sf'] = Sf
        PRO['c'] = curve

        return PRO