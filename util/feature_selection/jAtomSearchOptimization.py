import numpy as np

from losses.jFitnessFunction import jFitnessFunction

class jAtomSearchOptimization:
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

        # Number of dimensions
        dim = x_train.shape[1]
        N = self.N
        # Initial
        X = np.zeros((N, dim))
        for i in range(N):
            for d in range(dim):
                X[i, d] = lb + (ub - lb) * np.random.rand()

        V = np.zeros((N, dim))
        for i in range(N):
            for d in range(dim):
                V[i, d] = lb + (ub - lb) * np.random.rand()

        # Pre
        temp_A = np.zeros((N, dim))
        fitG = np.inf
        fit = np.zeros(N)

        curve = np.zeros(self.max_Iter)
        t = 1

        # Iteration
        while t <= self.max_Iter:
            for i in range(N):
                # Fitness
                fit[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)
                # Best update
                if fit[i] < fitG:
                    fitG = fit[i]
                    Xbest = X[i, :]

            # Best & worst fitness (28-29)
            fitB = np.min(fit)
            fitW = np.max(fit)

            # Number of K neighbor (32)
            Kbest = int(np.ceil(N - (N - 2) * np.sqrt(t / self.max_Iter)))

            # Mass (26)
            M = np.exp(-(fit - fitB) / (fitW - fitB))
            # Normalized mass (27)
            M = M / np.sum(M)
            # Sort normalized mass in descending order
            idx_M = np.argsort(-M)

            # Constraint force (23-24)
            G = np.exp(-20 * t / self.max_Iter)
            E = np.zeros((N, dim))
            for i in range(N):
                XK = np.mean(X[idx_M[0:Kbest], :], axis=0)
                # Length scale (17)
                scale_dist = np.linalg.norm(X[i, :] - XK, ord=2)

                for ii in range(Kbest):
                    # Select neighbor with higher mass
                    j = idx_M[ii]
                    # Get LJ-potential
                    Po = self._LJ_Potential(X[i, :], X[j, :], t, self.max_Iter, scale_dist)
                    # Distance
                    dist = np.linalg.norm(X[i, :] - X[j, :], ord=2)
                    for d in range(dim):
                        # Update (25)
                        E[i, d] = E[i, d] + np.random.rand() * Po * ((X[j, d] - X[i, d]) / (dist + np.spacing(1)))

                for d in range(dim):
                    E[i, d] = self.alpha * E[i, d] + self.beta * (Xbest[d] - X[i, d])
                    # Calculate part of acceleration (25)
                    temp_A[i, d] = E[i, d] / M[i]

            # Update
            for i in range(N):
                for d in range(dim):
                    # Acceleration (25)
                    Acce = temp_A[i, d] * G
                    # Velocity update (30)
                    V[i, d] = np.random.rand() * V[i, d] + Acce
                    # Position update (31)
                    X[i, d] = X[i, d] + V[i, d]

                # Boundary
                XB = X[i, :]
                XB[XB > ub] = ub
                XB[XB < lb] = lb
                X[i, :] = XB

            curve[t-1] = fitG
            print('\nIteration %d Best (ASO)= %f' % (t, curve[t-1]))
            t = t + 1

        # Select features based on selected index
        Pos = np.arange(0, dim)
        Sf = Pos[(Xbest > self.thres) == 1]

        # Store results
        ASO = {}
        ASO['sf'] = Sf
        ASO['c'] = curve

        return ASO

    @staticmethod
    #// LJ-Potential //
    def _LJ_Potential(X1, X2, t, max_Iter, scale_dist):
        # Calculate LJ-potential
        h0 = 1.1
        u = 1.24
        # Equilibration distance [Assume 1.12*(17)~=(17)]
        r = np.linalg.norm(X1 - X2, ord=2)
        # Depth function (15)
        n = (1 - (t - 1) / max_Iter) ** 3
        # Drift factor (19)
        g = 0.1 * np.sin((np.pi / 2) * (t / max_Iter))
        # Hmax & Hmin (18)
        Hmin = h0 + g
        Hmax = u
        # Compute H (16)
        if r / scale_dist < Hmin:
            H = Hmin
        elif r / scale_dist > Hmax:
            H = Hmax
        else:
            H = r / scale_dist
        # Revised version (14,25)
        Potential = n * (12 * (-H) ** (-13) - 6 * (-H) ** (-7))
        return Potential

