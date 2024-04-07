import numpy as np

from losses.jFitnessFunction import jFitnessFunction

class jHarrisHawksOptimization:
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

        # Pre
        fit = np.zeros(N)
        fitR = np.inf
        Y = np.zeros(dim)
        Z = np.zeros(dim)

        curve = np.zeros(self.max_Iter)
        t = 1

        # Iterations
        while t <= self.max_Iter:
            for i in range(N):
                # Fitness
                fit[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)

                # Rabbit update
                if fit[i] < fitR:
                    fitR = fit[i]
                    Xrb = X[i, :]

            # Mean position of hawk (2)
            X_mu = np.mean(X, axis=0)

            for i in range(N):
                # Random number in [-1,1]
                E0 = -1 + 2 * np.random.rand()
                # Escaping energy of rabbit (3)
                E = 2 * E0 * (1 - (t / self.max_Iter))

                # Exploration phase
                if abs(E) >= 1:
                    # Define q in [0,1]
                    q = np.random.rand()

                    if q >= 0.5:
                        # Random select a hawk k
                        k = np.random.randint(0, N)
                        r1 = np.random.rand()
                        r2 = np.random.rand()

                        for d in range(dim):
                            # Position update (1)
                            X[i, d] = X[k, d] - r1 * abs(X[k, d] - 2 * r2 * X[i, d])

                    elif q < 0.5:
                        r3 = np.random.rand()
                        r4 = np.random.rand()

                        for d in range(dim):
                            # Update Hawk (1)
                            X[i, d] = (Xrb[d] - X_mu[d]) - r3 * (lb + r4 * (ub - lb))

                # Exploitation phase
                elif abs(E) < 1:
                    # Jump strength
                    J = 2 * (1 - np.random.rand())
                    r = np.random.rand()

                    # {1} Soft besiege
                    if r >= 0.5 and abs(E) >= 0.5:
                        for d in range(dim):
                            # Delta X (5)
                            DX = Xrb[d] - X[i, d]
                            # Position update (4)
                            X[i, d] = DX - E * abs(J * Xrb[d] - X[i, d])

                    # {2} hard besiege
                    elif r >= 0.5 and abs(E) < 0.5:
                        for d in range(dim):
                            # Delta X (5)
                            DX = Xrb[d] - X[i, d]
                            # Position update (6)
                            X[i, d] = Xrb[d] - E * abs(DX)

                    # {3} Soft besiege with progressive rapid dives
                    elif r < 0.5 and abs(E) >= 0.5:
                        # Levy distribution (9)
                        LF = self._Levy_Distribution(self.beta, dim)

                        for d in range(dim):
                            # Compute Y (7)
                            Y[d] = Xrb[d] - E * abs(J * Xrb[d] - X[i, d])
                            # Compute Z (8)
                            Z[d] = Y[d] + np.random.rand() * LF[d]

                        # Boundary
                        Y[Y > ub] = ub
                        Y[Y < lb] = lb
                        Z[Z > ub] = ub
                        Z[Z < lb] = lb

                        # Fitness
                        fitY = self.loss_func(x_train[:, Y > self.thres], x_test[:, Y > self.thres], y_train, y_test)
                        fitZ = self.loss_func(x_train[:, Z > self.thres], x_test[:, Z > self.thres], y_train, y_test)

                        # Greedy selection (10)
                        if fitY < fit[i]:
                            fit[i] = fitY
                            X[i, :] = Y
                        if fitZ < fit[i]:
                            fit[i] = fitZ
                            X[i, :] = Z

                    # {4} Hard besiege with progressive rapid dives
                    elif r < 0.5 and abs(E) < 0.5:
                        # Levy distribution (9)
                        LF = self._Levy_Distribution(self.beta, dim)

                        for d in range(dim):
                            # Compute Y (12)
                            Y[d] = Xrb[d] - E * abs(J * Xrb[d] - X_mu[d])
                            # Compute Z (13)
                            Z[d] = Y[d] + np.random.rand() * LF[d]

                        # Boundary
                        Y[Y > ub] = ub
                        Y[Y < lb] = lb
                        Z[Z > ub] = ub
                        Z[Z < lb] = lb

                        # Fitness
                        fitY = self.loss_func(x_train[:, Y > self.thres], x_test[:, Y > self.thres], y_train, y_test)
                        fitZ = self.loss_func(x_train[:, Z > self.thres], x_test[:, Z > self.thres], y_train, y_test)

                        # Greedy selection (11)
                        if fitY < fit[i]:
                            fit[i] = fitY
                            X[i, :] = Y
                        if fitZ < fit[i]:
                            fit[i] = fitZ
                            X[i, :] = Z

                # Boundary
                XB = X[i, :]
                XB[XB > ub] = ub
                XB[XB < lb] = lb
                X[i, :] = XB

            # Save
            curve[t - 1] = fitR
            print(f'\nIteration {t} Best (HHO) = {curve[t - 1]}')
            t = t + 1

        # Select features
        Pos = np.arange(1, dim + 1)
        Sf = Pos[(Xrb > self.thres) == 1]

        # Store results
        HHO = {}
        HHO['sf'] = Sf
        HHO['c'] = curve

        return HHO

    @staticmethod
    def _Levy_Distribution(beta, dim):
        # Sigma
        nume = np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
        deno = np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
        sigma = (nume / deno) ** (1 / beta)

        # Parameter u & v
        u = np.random.randn(dim) * sigma
        v = np.random.randn(dim)

        # Step
        step = u / abs(v) ** (1 / beta)
        LF = 0.01 * step

        return LF
