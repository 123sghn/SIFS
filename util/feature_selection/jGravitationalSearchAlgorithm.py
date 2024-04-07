import numpy as np

from losses.jFitnessFunction import jFitnessFunction

class jGravitationalSearchAlgorithm:
    def __init__(self, N, max_Iter, loss_func, alpha=0.9, beta=0.1, thres=0.5, tau=1, rho=0.2, eta=1, G0=100):
        self.N = N
        self.max_Iter = max_Iter
        self.loss_func = loss_func
        self.alpha = alpha
        self.beta = beta
        self.thres = thres
        self.tau = tau
        self.rho = rho
        self.eta = eta
        self.G0 = G0

    def optimize(self, x_train, x_test, y_train, y_test):

        # Parameters
        lb = 0
        ub = 1
        G0 = self.G0   # initial gravitational constant
        N = self.N

        # Number of dimensions
        dim = x_train.shape[1]
        # Initial population
        X = np.zeros((N, dim))
        V = np.zeros((N, dim))
        for i in range(N):
            for d in range(dim):
                X[i, d] = lb + (ub - lb) * np.random.rand()
        # Pre
        fit = np.zeros(N)
        fitG = np.inf

        curve = np.zeros(self.max_Iter)
        t = 1
        # Iteration
        while t <= self.max_Iter:
            for i in range(N):
                # Fitness
                fit[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)
                # Global best
                if fit[i] < fitG:
                    Xgb = X[i, :]
                    fitG = fit[i]
            # The best & the worst fitness (17-18)
            best = np.min(fit)
            worst = np.max(fit)
            # Normalization mass (15)
            mass = (fit - worst) / (best - worst)
            # Compute inertia mass (16)
            M = mass / np.sum(mass)
            # Update gravitaty constant (28)
            G = G0 * np.exp(-self.alpha * (t / self.max_Iter))
            # Kbest linearly decreases from N to 1
            Kbest = round(N - (N - 1) * (t / self.max_Iter))
            # Sort mass in descending order
            idx_M = np.argsort(M)[::-1]
            E = np.zeros((N, dim))
            for i in range(N):
                for ii in range(Kbest):
                    j = idx_M[ii]
                    if j != i:
                        # Euclidean distance (8)
                        R = np.sqrt(np.sum((X[i, :] - X[j, :]) ** 2))
                        for d in range(dim):
                            # Note that Mp(i)/M(i)=1 (7,9)
                            E[i, d] = E[i, d] + np.random.rand() * M[j] * ((X[j, d] - X[i, d]) / (R + np.finfo(float).eps))
            # Search agent update
            for i in range(N):
                for d in range(dim):
                    # Acceleration: Note Mii(t) ~1 (10)
                    Acce = E[i, d] * G
                    # Velocity update (11)
                    V[i, d] = np.random.rand() * V[i, d] + Acce
                    # Position update (12)
                    X[i, d] = X[i, d] + V[i, d]
                # Boundary
                XB = X[i, :]
                XB[XB > ub] = ub
                XB[XB < lb] = lb
                X[i, :] = XB
            curve[t - 1] = fitG
            print('\nIteration {} Best (GSA)= {}'.format(t, curve[t - 1]))
            t = t + 1
        # Select features
        Pos = np.arange(dim)
        Sf = Pos[(Xgb > self.thres) == 1]
        # Store results
        GSA = {}
        GSA['sf'] = Sf
        GSA['c'] = curve

        return GSA