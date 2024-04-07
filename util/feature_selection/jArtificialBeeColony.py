import numpy as np

from losses.jFitnessFunction import jFitnessFunction


class jArtificialBeeColony:
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
        max_limit=5,
    ):
        self.N = N
        self.max_Iter = max_Iter
        self.loss_func = loss_func
        self.alpha = alpha
        self.beta = beta
        self.thres = thres
        self.tau = tau
        self.rho = rho
        self.eta = eta
        self.max_limit = max_limit

    def optimize(self, x_train, x_test, y_train, y_test):
        # Parameters
        lb = 0
        ub = 1
        max_limit = self.max_limit

        # Number of dimensions
        dim = x_train.shape[1]
        # Divide into employ and onlooker bees
        N = self.N // 2
        # Initial
        X = np.zeros((N, dim))
        for i in range(N):
            for d in range(dim):
                X[i, d] = lb + (ub - lb) * np.random.rand()
        # Fitness (9)
        fit = np.zeros(N)
        fitG = np.inf
        for i in range(N):
            fit[i] = self.loss_func(
                x_train[:, X[i, :] > self.thres],
                x_test[:, X[i, :] > self.thres],
                y_train,
                y_test,
            )
            # Best food source
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i, :]
        # Pre
        limit = np.zeros(N)
        V = np.zeros((N, dim))

        curve = np.zeros(self.max_Iter)
        curve[0] = fitG
        t = 2
        # Iteration
        while t <= self.max_Iter:
            # {1} Employed bee phase
            for i in range(N):
                # Choose k randomly, but not equal to i
                k = list(range(i)) + list(range(i + 1, N))
                k = k[np.random.randint(0, len(k))]
                for d in range(dim):
                    # Phi in [-1,1]
                    phi = -1 + 2 * np.random.rand()
                    # Position update (6)
                    V[i, d] = X[i, d] + phi * (X[i, d] - X[k, d])
                # Boundary
                XB = V[i, :]
                XB[XB > ub] = ub
                XB[XB < lb] = lb
                V[i, :] = XB
            # Fitness
            for i in range(N):
                # Fitness
                Fnew = self.loss_func(
                    x_train[:, V[i, :] > self.thres],
                    x_test[:, V[i, :] > self.thres],
                    y_train,
                    y_test,
                )
                # Compare neighbor bee
                if Fnew <= fit[i]:
                    # Update bee & reset limit counter
                    X[i, :] = V[i, :]
                    fit[i] = Fnew
                    limit[i] = 0
                else:
                    # Update limit counter
                    limit[i] += 1
            # Minimization problem (5)
            Ifit = 1 / (1 + fit)
            # Convert probability (7)
            prob = Ifit / np.sum(Ifit)

            # {2} Onlooker bee phase
            i = 0
            m = 0
            while m < N:
                if np.random.rand() < prob[i]:
                    # Choose k randomly, but not equal to i
                    k = list(range(i)) + list(range(i + 1, N))
                    k = k[np.random.randint(0, len(k))]
                    for d in range(dim):
                        # Phi in [-1,1]
                        phi = -1 + 2 * np.random.rand()
                        # Position update (6)
                        V[i, d] = X[i, d] + phi * (X[i, d] - X[k, d])
                    # Boundary
                    XB = V[i, :]
                    XB[XB > ub] = ub
                    XB[XB < lb] = lb
                    V[i, :] = XB
                    # Fitness
                    Fnew = self.loss_func(
                        x_train[:, V[i, :] > self.thres],
                        x_test[:, V[i, :] > self.thres],
                        y_train,
                        y_test,
                    )
                    # Greedy selection
                    if Fnew <= fit[i]:
                        X[i, :] = V[i, :]
                        fit[i] = Fnew
                        limit[i] = 0
                        # Re-compute new probability (5,7)
                        Ifit = 1 / (1 + fit)
                        prob = Ifit / np.sum(Ifit)
                    else:
                        limit[i] += 1
                    m += 1
                # Reset i
                i += 1
                if i >= N:
                    i = 0

            # {3} Scout bee phase
            for i in range(N):
                if limit[i] >= max_limit:
                    for d in range(dim):
                        # Produce new bee (8)
                        X[i, d] = lb + (ub - lb) * np.random.rand()
                    # Reset Limit
                    limit[i] = 0
                    # Fitness
                    fit[i] = self.loss_func(
                        x_train[:, X[i, :] > self.thres],
                        x_test[:, X[i, :] > self.thres],
                        y_train,
                        y_test,
                    )
                # Best food source
                if fit[i] < fitG:
                    fitG = fit[i]
                    Xgb = X[i, :]
            curve[t - 1] = fitG
            print("\nIteration %d Best (ABC)= %f" % (t, curve[t - 1]))
            t += 1
        # Select features based on selected index
        Pos = np.arange(dim)
        Sf = Pos[Xgb > self.thres]
        # Store results
        ABC = {"sf": Sf.tolist(), "c": curve.tolist()}
        return ABC
