import numpy as np
from scipy.special import gamma
from losses.jFitnessFunction import jFitnessFunction

class jFlowerPollinationAlgorithm:
    def __init__(self, N, max_Iter, loss_func, alpha=0.9, beta=0.1, thres=0.5, tau=1, rho=0.2, eta=1, P=0.8):
        self.N = N
        self.max_Iter = max_Iter
        self.loss_func = loss_func
        self.alpha = alpha
        self.beta = beta
        self.thres = thres
        self.tau = tau
        self.rho = rho
        self.eta = eta
        self.P = P

    def optimize(self, x_train, x_test, y_train, y_test):

        # Parameters
        lb = 0
        ub = 1
        P = self.P  # switch probability
        N = self.N

        # Number of dimensions
        dim = x_train.shape[1]
        # Initial
        X = np.zeros((N, dim))
        for i in range(N):
            for d in range(dim):
                X[i, d] = lb + (ub - lb) * np.random.rand()

        # Compute fitness
        fit = np.zeros(N)
        fitG = float('inf')
        for i in range(N):
            fit[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)
            # Best flower
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i, :]

        # Pre
        Xnew = np.zeros((N, dim))

        curve = np.zeros(self.max_Iter)
        curve[0] = fitG
        t = 2
        # Iterations
        while t <= self.max_Iter:
            for i in range(N):
                # Global pollination
                if np.random.rand() < P:
                    # Levy distribution (2)
                    L = self._levy_distribution(self.beta, dim)
                    for d in range(dim):
                        # Global pollination (1)
                        Xnew[i, d] = X[i, d] + L[d] * (X[i, d] - Xgb[d])
                # Local pollination
                else:
                    # Different flower j, k in same species
                    R = np.random.permutation(N)
                    J = R[0]
                    K = R[1]
                    # Epsilon [0 to 1]
                    eps = np.random.rand()
                    for d in range(dim):
                        # Local pollination (3)
                        Xnew[i, d] = X[i, d] + eps * (X[J, d] - X[K, d])

                # Check boundary
                XB = Xnew[i, :]
                XB[XB > ub] = ub
                XB[XB < lb] = lb
                Xnew[i, :] = XB

            # Fitness
            for i in range(N):
                # Compute fitness
                Fnew = self.loss_func(x_train[:, Xnew[i, :] > self.thres], x_test[:, Xnew[i, :] > self.thres],
                                        y_train, y_test)
                # Update if there is better solution
                if Fnew <= fit[i]:
                    X[i, :] = Xnew[i, :]
                    fit[i] = Fnew
                # Best flower
                if fit[i] < fitG:
                    Xgb = X[i, :]
                    fitG = fit[i]

            curve[t - 1] = fitG
            print('\nIteration %d Best (FPA)= %f' % (t, curve[t - 1]))
            t += 1

        # Select features
        Pos = np.arange(dim)
        Sf = Pos[(Xgb > self.thres) == 1]

        # Store results
        FPA = {}
        FPA['sf'] = Sf
        FPA['c'] = curve

        return FPA

    @staticmethod
    def _levy_distribution(beta, dim):
        # Sigma
        nume = gamma(1 + beta) * np.sin(np.pi * beta / 2)
        deno = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
        sigma = (nume / deno) ** (1 / beta)

        # Parameter u & v
        u = np.random.randn(dim) * sigma
        v = np.random.randn(dim)

        # Step
        step = u / np.abs(v) ** (1 / beta)

        LF = 0.01 * step
        return LF
