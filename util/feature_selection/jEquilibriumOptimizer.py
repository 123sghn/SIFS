import numpy as np

from losses.jFitnessFunction import jFitnessFunction

class jEquilibriumOptimizer:
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
        a1 = 2  # constant
        a2 = 1  # constant
        GP = 0.5  # generation probability
        V = 1  # unit
        N = self.N

        # Number of dimensions
        dim = x_train.shape[1]
        # Initial (6)
        X = np.zeros((N, dim))
        for i in range(N):
            for d in range(dim):
                X[i, d] = lb + (ub - lb) * np.random.rand()

        # Set memory
        Xmb = np.zeros((N, dim))
        fitM = np.ones(N)
        # Pre
        fitE1 = np.inf
        fitE2 = np.inf
        fitE3 = np.inf
        fitE4 = np.inf
        Xeq1 = np.zeros(dim)
        Xeq2 = np.zeros(dim)
        Xeq3 = np.zeros(dim)
        Xeq4 = np.zeros(dim)
        Xave = np.zeros(dim)
        fit = np.zeros(N)
        curve = np.zeros(self.max_Iter)
        t = 1

        # Iteration
        while t <= self.max_Iter:
            # Fitness
            for i in range(N):
                fit[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)
                # Equilibrium update
                if fit[i] < fitE1:
                    fitE1 = fit[i]
                    Xeq1 = X[i, :]
                elif fit[i] > fitE1 and fit[i] < fitE2:
                    fitE2 = fit[i]
                    Xeq2 = X[i, :]
                elif fit[i] > fitE1 and fit[i] > fitE2 and fit[i] < fitE3:
                    fitE3 = fit[i]
                    Xeq3 = X[i, :]
                elif fit[i] > fitE1 and fit[i] > fitE2 and fit[i] > fitE3 and fit[i] < fitE4:
                    fitE4 = fit[i]
                    Xeq4 = X[i, :]
            # Memory update
            for i in range(N):
                if fitM[i] < fit[i]:
                    fit[i] = fitM[i]
                    X[i, :] = Xmb[i, :]
            # Store memory
            Xmb = X.copy()
            fitM = fit.copy()
            # Compute average candidate
            for d in range(dim):
                Xave[d] = (Xeq1[d] + Xeq2[d] + Xeq3[d] + Xeq4[d]) / 4
            # Make an equilibrium pool (7)
            Xpool = np.vstack((Xeq1, Xeq2, Xeq3, Xeq4, Xave))
            # Compute function tt (9)
            T = (1 - (t / self.max_Iter)) ** (a2 * (t / self.max_Iter))
            # Update
            for i in range(N):
                # Generation rate control parameter (15)
                r1 = np.random.rand()
                r2 = np.random.rand()
                if r2 >= GP:
                    GCP = 0.5 * r1
                else:
                    GCP = 0
                # Random one solution from Xpool
                eq = np.random.randint(0, 5)
                for d in range(dim):
                    # Random in [0,1]
                    r = np.random.rand()
                    # Lambda in [0,1]
                    lambd = np.random.rand()
                    # Substitution (11)
                    F = a1 * np.sign(r - 0.5) * (np.exp(-lambd * T) - 1)
                    # Compute G0 (14)
                    G0 = GCP * (Xpool[eq, d] - lambd * X[i, d])
                    # Compute G (13)
                    G = G0 * F
                    # Update (16)
                    X[i, d] = Xpool[eq, d] + (X[i, d] - Xpool[eq, d]) * F + (G / (lambd * V)) * (1 - F)
                # Boundary
                XB = X[i, :]
                XB[XB > ub] = ub
                XB[XB < lb] = lb
                X[i, :] = XB
            curve[t-1] = fitE1
            print('\nIteration {} Best (EO)= {}'.format(t, curve[t-1]))
            t += 1

        # Select features based on selected index
        Pos = np.arange(dim)
        Sf = Pos[Xeq1 > self.thres]

        # Store results
        EO = {}
        EO['sf'] = Sf
        EO['c'] = curve

        return EO
