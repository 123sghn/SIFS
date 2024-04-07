import numpy as np
from scipy.special import gamma

from losses.jFitnessFunction import jFitnessFunction


class jCuckooSearchAlgorithm:
    def __init__(
        self,
        N,
        max_Iter,
        loss_func,
        alpha=0.9,
        beta=0.1,
        thres=0.5,
        tau=1,
        rho=0.2,
        eta=1,
        Pa=0.25,
    ):
        self.loss_func = loss_func
        self.tau = tau
        self.eta = eta
        self.max_Iter = max_Iter
        self.N = N
        self.thres = thres
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Pa = Pa

    def optimize(self, x_train, x_test, y_train, y_test):
        # Parameters
        lb = 0
        ub = 1
        Pa = self.Pa
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
            fit[i] = self.loss_func(
                x_train[:, X[i, :] > self.thres],
                x_test[:, X[i, :] > self.thres],
                y_train,
                y_test,
            )
            # Best cuckoo nest
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
            # {1} Random walk/Levy flight phase
            for i in range(N):
                # Levy distribution
                L = self._levy_distribution(self.beta, dim)
                for d in range(dim):
                    # Levy flight (1)
                    Xnew[i][d] = X[i][d] + self.alpha * L[d] * (X[i][d] - Xgb[d])
                # Boundary
                XB = Xnew[i, :]
                XB[XB > ub] = ub
                XB[XB < lb] = lb
                Xnew[i, :] = XB
            # Fintess
            for i in range(N):
                # Fitness
                Fnew = self.loss_func(
                    x_train[:, Xnew[i, :] > self.thres],
                    x_test[:, Xnew[i, :] > self.thres],
                    y_train,
                    y_test,
                )
                # Greedy selection
                if Fnew <= fit[i]:
                    fit[i] = Fnew
                    X[i, :] = Xnew[i, :]
            # {2} Discovery and abandon worse nests phase
            Xj = X[np.random.permutation(N), :]
            Xk = X[np.random.permutation(N), :]
            for i in range(N):
                Xnew[i, :] = X[i, :]
                r = np.random.rand()
                for d in range(dim):
                    # A fraction of worse nest is discovered with a probability
                    if np.random.rand() < Pa:
                        Xnew[i, d] = X[i, d] + r * (Xj[i, d] - Xk[i, d])
                # Boundary
                XB = Xnew[i, :]
                XB[XB > ub] = ub
                XB[XB < lb] = lb
                Xnew[i, :] = XB
            # Fitness
            for i in range(N):
                # Fitness
                Fnew = self.loss_func(
                    x_train[:, Xnew[i, :] > self.thres],
                    x_test[:, Xnew[i, :] > self.thres],
                    y_train,
                    y_test,
                )
                # Greedy selection
                if Fnew <= fit[i]:
                    fit[i] = Fnew
                    X[i, :] = Xnew[i, :]
                # Best cuckoo
                if fit[i] < fitG:
                    fitG = fit[i]
                    Xgb = X[i, :]
            curve[t - 1] = fitG
            print("\nIteration {} Best (CS) = {}".format(t, curve[t - 1]))
            t += 1
        # Select features
        Pos = np.arange(dim)
        Sf = Pos[Xgb > self.thres]
        # Store results
        CS = {
            "sf": Sf,
            "c": curve,
        }
        return CS

    # // Levy Flight //
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
