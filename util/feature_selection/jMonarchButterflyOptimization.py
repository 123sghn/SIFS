import numpy as np

from losses.jFitnessFunction import jFitnessFunction
from scipy.special import gamma

class jMonarchButterflyOptimization:
    def __init__(self, N, max_Iter, loss_func, alpha=0.9, beta=0.1, thres=0.5, tau=1, rho=0.2, eta=1,peri=1.2, p=0.5, Smax=1,BAR=0.5,num_land1=4):
        self.N = N
        self.max_Iter = max_Iter
        self.loss_func = loss_func
        self.alpha = alpha
        self.beta = beta
        self.thres = thres
        self.tau = tau
        self.rho = rho
        self.eta = eta
        self.peri = peri
        self.p=p
        self.Smax=Smax
        self.BAR=BAR
        self.num_land1=num_land1

    def optimize(self, x_train, x_test, y_train, y_test):

        # Parameters
        lb = 0
        ub = 1
        peri = self.peri  # migration period
        p = self.p  # ratio
        Smax = self.Smax  # maximum step
        BAR = self.BAR  # butterfly adjusting rate
        num_land1 = self.num_land1  # number of butterflies in land 1

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
            # Global best update
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i, :]
        # Pre
        Xnew = np.zeros((N, dim))
        Fnew = np.zeros(N)

        curve = np.zeros(self.max_Iter)
        curve[0] = fitG
        t = 1
        # Iterations
        while t < self.max_Iter:
            # Sort butterfly
            idx = np.argsort(fit)
            fit = fit[idx]
            X = X[idx, :]
            # Weight factor (8)
            alpha = Smax / (t ** 2)
            # {1} First land: Migration operation
            for i in range(num_land1):
                for d in range(dim):
                    # Random number (2)
                    r = np.random.rand() * peri
                    if r <= p:
                        # Random select a butterfly in land 1
                        r1 = np.random.randint(0, num_land1)
                        # Update position (1)
                        Xnew[i, d] = X[r1, d]
                    else:
                        # Random select a butterfly in land 2
                        r2 = np.random.randint(num_land1, N)
                        # Update position (3)
                        Xnew[i, d] = X[r2, d]
                # Boundary
                XB = Xnew[i, :]
                XB[XB > ub] = ub
                XB[XB < lb] = lb
                Xnew[i, :] = XB

            # {2} Second land: Butterly adjusting operation
            for i in range(num_land1, N):
                # Levy distribution (7)
                dx = self._Levy_Distribution(self.beta, dim)
                for d in range(dim):
                    if np.random.rand() <= p:
                        # Position update (4)
                        Xnew[i, d] = Xgb[d]
                    else:
                        # Random select a butterfly in land 2
                        r3 = np.random.randint(num_land1, N)
                        # Update position (5)
                        Xnew[i, d] = X[r3, d]
                        # Butterfly adjusting (6)
                        if np.random.rand() > BAR:
                            Xnew[i, d] = Xnew[i, d] + alpha * (dx[d] - 0.5)
                # Boundary
                XB = Xnew[i, :]
                XB[XB > ub] = ub
                XB[XB < lb] = lb
                Xnew[i, :] = XB

            # {3} Combine population
            for i in range(N):
                # Fitness
                Fnew[i] = self.loss_func(x_train[:, Xnew[i, :] > self.thres], x_test[:, Xnew[i, :] > self.thres],
                                        y_train, y_test)
                # Global best update
                if Fnew[i] < fitG:
                    fitG = Fnew[i]
                    Xgb = Xnew[i, :]
            # Merge & Select best N solutions
            XX = np.concatenate((X, Xnew), axis=0)
            FF = np.concatenate((fit, Fnew))
            idx = np.argsort(FF)
            X = XX[idx[:N], :]
            fit = FF[:N]
            # Save
            curve[t] = fitG
            print('\nIteration %d Best (MBO)= %f' % (t, curve[t]))
            t += 1
        # Select features
        Pos = np.arange(dim)
        Sf = Pos[(Xgb > self.thres) == 1]
        # Store results
        MBO = {}
        MBO['sf'] = Sf
        MBO['c'] = curve

        return MBO

    @staticmethod
    # // Levy Flight //
    def _Levy_Distribution(beta, dim):
        # Sigma
        nume = gamma(1 + beta) * np.sin(np.pi * beta / 2)
        deno = gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)
        sigma = (nume / deno)**(1 / beta)
        # Parameter u & v
        u = np.random.randn(dim) * sigma
        v = np.random.randn(dim)
        # Step
        step = u / np.abs(v)**(1 / beta)
        LF = step
        return LF
