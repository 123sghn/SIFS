import numpy as np

from losses.jFitnessFunction import jFitnessFunction

class jBatAlgorithm:
    def __init__(self, N, max_Iter, loss_func, alpha=0.9, beta=0.1, thres=0.5, tau=1, rho=0.2, eta=1, gamma=0.9, A_max=2, r0_max=1):
        self.N = N
        self.max_Iter = max_Iter
        self.loss_func = loss_func
        self.alpha = alpha
        self.beta = beta
        self.thres = thres
        self.tau = tau
        self.rho = rho
        self.eta = eta
        self.gamma=gamma
        self.A_max=A_max
        self.r0_max=r0_max

    def optimize(self, x_train, x_test, y_train, y_test):
        # Parameters
        lb = 0
        ub = 1
        fmax = 2     # maximum frequency
        fmin = 0     # minimum frequency
        gamma = self.gamma   # constant
        A_max  = self.A_max     # maximum loudness
        r0_max = self.r0_max     # maximum pulse rate
        N = self.N

        # Number of dimensions
        dim = np.size(x_train, 1)
        # Initial
        X = np.zeros((N, dim))
        V = np.zeros((N, dim))
        for i in range(N):
            for d in range(dim):
                X[i, d] = lb + (ub - lb) * np.random.rand()

        # Fitness
        fit = np.zeros(N)
        fitG = np.inf
        for i in range(N):
            fit[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)
            # Global best
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i, :]

        # Loudness of each bat, [1 ~ 2]
        A = np.random.uniform(1, A_max, (N, 1))
        # Pulse rate of each bat, [0 ~ 1]
        r0 = np.random.uniform(0, r0_max, (N, 1))
        r = r0

        # Pre
        Xnew = np.zeros((N, dim))

        curve = np.zeros(self.max_Iter)
        curve[0] = fitG
        t = 2

        # Iterations
        while t <= self.max_Iter:
            for i in range(N):
                # Beta [0~1]
                beta = np.random.rand()
                # Frequency (2)
                freq = fmin + (fmax - fmin) * beta

                for d in range(dim):
                    # Velocity update (3)
                    V[i, d] = V[i, d] + (X[i, d] - Xgb[d]) * freq
                    # Position update (4)
                    Xnew[i, d] = X[i, d] + V[i, d]

                # Generate local solution around best solution
                if np.random.rand() > r[i]:
                    for d in range(dim):
                        # Epsilon in [-1,1]
                        eps = -1 + 2 * np.random.rand()
                        # Random walk (5)
                        Xnew[i, d] = Xgb[d] + eps * np.mean(A)

                # Boundary
                XB = Xnew[i, :]
                XB[XB > ub] = ub
                XB[XB < lb] = lb
                Xnew[i, :] = XB

            # Fitness
            for i in range(N):
                # Fitness
                Fnew = self.loss_func(x_train[:, Xnew[i, :] > self.thres], x_test[:, Xnew[i, :] > self.thres],
                                        y_train, y_test)
                # Greedy selection
                if np.random.rand() < A[i] and Fnew <= fit[i]:
                    X[i, :] = Xnew[i, :]
                    fit[i] = Fnew
                    # Loudness update (6)
                    A[i] = self.alpha * A[i]
                    # Pulse rate update (6)
                    r[i] = r0[i] * (1 - np.exp(-gamma * t))

                # Global best
                if fit[i] < fitG:
                    fitG = fit[i]
                    Xgb = X[i, :]

            curve[t - 1] = fitG
            print('\nIteration', t, 'Best (BA)=', curve[t - 1])
            t = t + 1

        # Select features
        Pos = np.arange(0, dim)
        Sf = Pos[(Xgb > self.thres) == 1]

        # Store results
        BA = {'sf': Sf, 'c': curve }
        return BA