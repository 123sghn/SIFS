import numpy as np

from losses.jFitnessFunction import jFitnessFunction

class jSlimeMouldAlgorithm:
    def __init__(self, N, max_Iter, loss_func, alpha=0.9, beta=0.1, thres=0.5, tau=1, rho=0.2, eta=1, z=0.03):
        self.N = N
        self.max_Iter = max_Iter
        self.loss_func = loss_func
        self.alpha = alpha
        self.beta = beta
        self.thres = thres
        self.tau = tau
        self.rho = rho
        self.eta = eta
        self.z = z

    def optimize(self, x_train, x_test, y_train, y_test):

        # Parameters
        lb = 0
        ub = 1
        z = self.z  # control local & global
        N = self.N
        dim = x_train.shape[1]

        # Initial
        X = np.zeros((N, dim))
        for i in range(N):
            for d in range(dim):
                X[i, d] = lb + (ub - lb) * np.random.rand()

        # Pre
        fit = np.zeros(N)
        fitG = np.inf
        W = np.zeros((N, dim))

        curve = np.zeros(self.max_Iter)
        t = 0

        # Iteration
        while t < self.max_Iter:
            # Fitness
            for i in range(N):

                fit[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)
                # Best
                if fit[i] < fitG:
                    fitG = fit[i]
                    Xb = X[i, :].copy()

            # Sort smell index (2.6)
            idxS = np.argsort(fit)
            fitS = fit[idxS]

            # Best fitness & worst fitness
            bF = np.min(fit)
            wF = np.max(fit)

            # Compute W (2.5)
            for i in range(N):
                for d in range(dim):
                    # Condition
                    r = np.random.rand()
                    if i <= N / 2:
                        W[idxS[i], d] = 1 + r * np.log10(((bF - fitS[i]) / (bF - wF + np.finfo(float).eps)) + 1)
                    else:
                        W[idxS[i], d] = 1 - r * np.log10(((bF - fitS[i]) / (bF - wF + np.finfo(float).eps)) + 1)

            # Compute a (2.4)
            a = np.arctanh(-((t+1) / self.max_Iter) + 1)

            # Compute b
            b = 1 - (t / self.max_Iter)

            # Update (2.7)
            for i in range(N):
                if np.random.rand() < z:
                    for d in range(dim):
                        X[i, d] = np.random.rand() * (ub - lb) + lb
                else:
                    # Update p (2.2)
                    p = np.tanh(np.abs(fit[i] - fitG))

                    # Update vb (2.3)


                    vb = np.random.uniform(-a, a, dim)

                    # Update vc
                    vc = np.random.uniform(-b, b,  dim)

                    for d in range(dim):
                        # Random in [0,1]
                        r = np.random.rand()

                        # Two random individuals
                        A = np.random.randint(0, N)
                        B = np.random.randint(0, N)

                        if r < p:
                            X[i][d] = Xb[d] + vb[d] * (W[i][d] * X[A][d] - X[B][d])
                        else:
                            X[i][d] = vc[d] * X[i][d]

                # Boundary
                X[i, :] = np.clip(X[i, :], lb, ub)

            # Save
            curve[t] = fitG
            print(f'\nIteration {t} Best (SMA)= {curve[t]}')
            t += 1

        # Select features based on selected index
        Pos = np.arange(dim)
        Sf = Pos[(Xb > self.thres) == True]

        # Store results
        SMA = {}
        SMA['sf'] = Sf
        SMA['c'] = curve

        return SMA