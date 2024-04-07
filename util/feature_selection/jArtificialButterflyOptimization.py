import numpy as np

from losses.jFitnessFunction import jFitnessFunction

class jArtificialButterflyOptimization:
    def __init__(self, N, max_Iter, loss_func, alpha=0.9, beta=0.1, thres=0.5, tau=1, rho=0.2, eta=1, step_e=0.05, ratio=0.2, type=1):
        self.N = N
        self.max_Iter = max_Iter
        self.loss_func = loss_func
        self.alpha = alpha
        self.beta = beta
        self.thres = thres
        self.tau = tau
        self.rho = rho
        self.eta = eta
        self.step_e = step_e
        self.ratio = ratio
        self.type = type

    def optimize(self, x_train, x_test, y_train, y_test):
        # Parameters
        lb = 0
        ub = 1
        step_e = self.step_e   # control number of sunspot
        ratio = self.ratio    # control step
        type = self.type      # type 1 or 2
        N = self.N
        # Number of dimensions
        dim = x_train.shape[1]
        # Initial
        X = np.zeros((self.N, dim))
        for i in range(N):
            for d in range(dim):
                X[i, d] = lb + (ub - lb) * np.random.rand()

        # Fitness
        fit = np.zeros(N)
        fitG = np.inf
        for i in range(N):
            fit[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)
            # Global update
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i, :]

        # Pre-processing
        Xnew = np.zeros((N, dim))

        curve = np.zeros(self.max_Iter)
        curve[0] = fitG
        t = 1
        # Iteration
        while t <= self.max_Iter:
            # Sort butterfly
            idx = np.argsort(fit)
            fit = fit[idx]
            X = X[idx, :]
            # Proportion of sunspot butterfly decreasing from 0.9 to ratio
            num_sun = round(N * (0.9 - (0.9 - ratio) * (t / self.max_Iter)))
            # Define a, linearly decrease from 2 to 0
            a = 2 - 2 * (t / self.max_Iter)
            # Step update (5)
            step = 1 - (1 - step_e) * (t / self.max_Iter)

            # {1} Some butterflies with better fitness: Sunspot butterfly
            for i in range(num_sun):
                # Random select a butterfly k, but not equal to i
                R = np.random.permutation(N)
                R = R[R != i]
                k = R[0]
                # [Version 1]
                if type == 1:
                    # Randomly select a dimension
                    J = np.random.randint(0, dim)
                    # Random number in [-1,1]
                    r1 = -1 + 2 * np.random.rand()
                    # Position update (1)
                    Xnew[i, :] = X[i, :]
                    Xnew[i, J] = X[i, J] + (X[i, J] - X[k, J]) * r1
                # [Version 2]
                elif type == 2:
                    # Distance
                    dist = np.linalg.norm(X[k, :] - X[i, :])
                    r2 = np.random.rand()
                    for d in range(dim):
                        # Position update (2)
                        Xnew[i, d] = X[i, d] + ((X[k, d] - X[i, d]) / dist) * (ub - lb) * step * r2
                # Boundary
                Xnew[i, :] = np.clip(Xnew[i, :], lb, ub)

            # Fitness
            for i in range(num_sun):
                # Fitness
                Fnew = self.loss_func(x_train[:, Xnew[i, :] > self.thres], x_test[:, Xnew[i, :] > self.thres],
                                        y_train, y_test)
                # Greedy selection
                if Fnew < fit[i]:
                    fit[i] = Fnew
                    X[i, :] = Xnew[i, :]
                # Global update
                if fit[i] < fitG:
                    fitG = fit[i]
                    Xgb = X[i, :]

            # {2} Some butterflies: Canopy butterfly
            for i in range(num_sun, N):
                # Random select a sunspot butterfly
                k = np.random.randint(0, num_sun)
                # [Version 1]
                if type == 1:
                    # Randomly select a dimension
                    J = np.random.randint(0, dim)
                    # Random number in [-1,1]
                    r1 = -1 + 2 * np.random.rand()
                    # Position update (1)
                    Xnew[i, :] = X[i, :]
                    Xnew[i, J] = X[i, J] + (X[i, J] - X[k, J]) * r1
                # [Version 2]
                elif type == 2:
                    # Distance
                    dist = np.linalg.norm(X[k, :] - X[i, :])
                    r2 = np.random.rand()
                    for d in range(dim):
                        # Position update (2)
                        Xnew[i, d] = X[i, d] + ((X[k, d] - X[i, d]) / dist) * (ub - lb) * step * r2
                # Boundary
                Xnew[i, :] = np.clip(Xnew[i, :], lb, ub)

            # Fitness
            for i in range(num_sun, N):
                # Fitness
                Fnew = self.loss_func(x_train[:, Xnew[i, :] > self.thres], x_test[:, Xnew[i, :] > self.thres],
                                        y_train, y_test)
                # Greedy selection
                if Fnew < fit[i]:
                    fit[i] = Fnew
                    X[i, :] = Xnew[i, :]
                else:
                    # Random select a butterfly
                    k = np.random.randint(0, N)
                    # Fly to new location
                    r3 = np.random.rand()
                    r4 = np.random.rand()
                    for d in range(dim):
                        # Compute D (4)
                        Dx = np.abs(2 * r3 * X[k, d] - X[i, d])
                        # Position update (3)
                        X[i, d] = X[k, d] - 2 * a * r4 - a * Dx
                    # Boundary
                    X[i, :] = np.clip(X[i, :], lb, ub)
                    # Fitness
                    fit[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)
                # Global update
                if fit[i] < fitG:
                    fitG = fit[i]
                    Xgb = X[i, :]

            curve[t - 1] = fitG
            if type == 1:
                print('\nIteration %d Best (ABO 1)= %f' % (t, curve[t - 1]))
            elif type == 2:
                print('\nIteration %d Best (ABO 2)= %f' % (t, curve[t - 1]))
            t = t + 1

        # Select features
        Pos = np.arange(1, dim + 1)
        Sf = Pos[(Xgb > self.thres)] - 1

        # Store results
        ABO = {}
        ABO['sf'] = Sf
        ABO['c'] = curve

        return ABO