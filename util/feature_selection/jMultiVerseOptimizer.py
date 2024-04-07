import numpy as np

from losses.jFitnessFunction import jFitnessFunction

class jMultiVerseOptimizer:
    def __init__(self, N, max_Iter, loss_func, alpha=0.9, beta=0.1, thres=0.5, tau=1, rho=0.2, eta=1,p=6,type=1):
        self.N = N
        self.max_Iter = max_Iter
        self.loss_func = loss_func
        self.alpha = alpha
        self.beta = beta
        self.thres = thres
        self.tau = tau
        self.rho = rho
        self.eta = eta
        self.p = p
        self.type = type

    def optimize(self, x_train, x_test, y_train, y_test):

        # Parameters
        lb = 0
        ub = 1
        p = self.p  # control TDR
        Wmax = 1  # maximum WEP
        Wmin = 0.2  # minimum WEP
        type = self.type
        N =self.N


        # Number of dimensions
        dim =x_train.shape[1]

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
            # Calculate inflation rate
            for i in range(N):
                fit[i] = self.loss_func(x_train[:,X[i, :] > self.thres], x_test[:,X[i, :] > self.thres], y_train, y_test)

                # Best universe
                if fit[i] < fitG:
                    fitG = fit[i]
                    Xgb = X[i, :]

            # Sort universe from best to worst
            idx = np.argsort(fit)
            fitSU = fit[idx]
            X_SU = X[idx, :]

            # Elitism (first 1 is elite)
            X[0, :] = X_SU[0, :]

            # Either 1-norm or 2-norm
            if type == 1:
                # Normalize inflation rate using 2-norm
                NI = fitSU / np.sqrt(np.sum(fitSU ** 2))
            elif type == 2:
                # Normalize inflation rate using 1-norm
                NI = fitSU / np.sum(fitSU)

            # Normalize inverse inflation rate using 1-norm
            inv_fitSU = 1 / (1 + fitSU)
            inv_NI = inv_fitSU / np.sum(inv_fitSU)

            # Wormhole Existence probability (3.3), increases from 0.2 to 1
            WEP = Wmin + t * ((Wmax - Wmin) / self.max_Iter)
            # Travelling disrance rate (3.4), descreases from 0.6 to 0
            TDR = 1 - ((t ** (1 / p)) / (self.max_Iter ** (1 / p)))

            # Start with 2 since first is elite
            for i in range(1, N):
                # Define black hole
                idx_BH = i
                for d in range(dim):
                    # White/black hole tunnels & exchange object of universes (3.1)
                    r1 = np.random.rand()
                    if r1 < NI[i]:
                        # Random select k with roulette wheel
                        idx_WH = self._Roulette_Wheel_Selection(inv_NI)
                        # Position update
                        X[idx_BH, d] = X_SU[idx_WH, d]

                    # Local changes for universes (3.2)
                    r2 = np.random.rand()
                    if r2 < WEP:
                        r3 = np.random.rand()
                        r4 = np.random.rand()
                        if r3 < 0.5:
                            X[i, d] = Xgb[d] + TDR * ((ub - lb) * r4 + lb)
                        else:
                            X[i, d] = Xgb[d] - TDR * ((ub - lb) * r4 + lb)
                    else:
                        X[i, d] = X[i, d]

                # Boundary
                XB = X[i, :]
                XB[XB > ub] = ub
                XB[XB < lb] = lb
                X[i, :] = XB

            curve[t-1] = fitG
            print('\nIteration %d Best (MVO)= %f' % (t, curve[t-1]))
            t += 1

        # Select features
        Pos = np.arange(dim)
        Sf = Pos[Xgb > self.thres]

        # Store results
        MVO = {'sf': Sf, 'c': curve}
        return MVO

    @staticmethod
    def _Roulette_Wheel_Selection(prob):
        # Cummulative summation
        C = np.cumsum(prob)
        # Random one value, most probability value [0~1]
        P = np.random.rand()
        # Route wheel
        for i in range(len(C)):
            if C[i] > P:
                return i