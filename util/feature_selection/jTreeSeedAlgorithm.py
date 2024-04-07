import numpy as np

from losses.jFitnessFunction import jFitnessFunction

class jTreeSeedAlgorithm:
    def __init__(self, N, max_Iter, loss_func, alpha=0.9, beta=0.1, thres=0.5, tau=1, rho=0.2, eta=1, ST=0.1):
        self.N = N
        self.max_Iter = max_Iter
        self.loss_func = loss_func
        self.alpha = alpha
        self.beta = beta
        self.thres = thres
        self.tau = tau
        self.rho = rho
        self.eta = eta
        self.ST=ST

    def optimize(self, x_train, x_test, y_train, y_test):

        # Parameters
        lb = 0
        ub = 1
        ST = self.ST  # switch probability
        N = self.N

        # Number of dimensions
        dim = x_train.shape[1]

        # Initial (5)
        X = np.zeros((N, dim))
        for i in range(N):
            for d in range(dim):
                X[i, d] = lb + (ub - lb) * np.random.rand()

        # Fitness
        fit = np.zeros(N)
        for i in range(N):
            fit[i] = self.loss_func(x_train[:,X[i, :] > self.thres], x_test[:,X[i, :] > self.thres], y_train, y_test)


        # Best solution (6)
        fitG = np.min(fit)
        idx = np.argmin(fit)
        Xgb = X[idx, :]

        # Maximum & minimum number of seed
        Smax = round(0.25 * N)
        Smin = round(0.1 * N)

        # Pre
        curve = np.zeros(self.max_Iter)
        curve[0] = fitG
        t = 2

        # Iteration
        while t <= self.max_Iter:
            for i in range(N):
                # Random number of seed
                num_seed = round(Smin + np.random.rand() * (Smax - Smin))
                Xnew = np.zeros((num_seed, dim))
                for j in range(num_seed):
                    # Random select a tree, but not i
                    RN = np.random.permutation(N)
                    RN = RN[RN != i]
                    r = RN[0]
                    for d in range(dim):
                        # Alpha in [-1,1]
                        alpha = -1 + 2 * np.random.rand()
                        if np.random.rand() < ST:
                            # Generate seed (3)
                            Xnew[j, d] = X[i, d] + alpha * (Xgb[d] - X[r, d])
                        else:
                            # Generate seed (4)
                            Xnew[j, d] = X[i, d] + alpha * (X[i, d] - X[r, d])
                    # Boundary
                    XB = Xnew[j, :]
                    XB[XB > ub] = ub
                    XB[XB < lb] = lb
                    Xnew[j, :] = XB

                # Fitness
                for j in range(num_seed):
                    # Fitness
                    Fnew = self.loss_func(x_train[:,Xnew[j, :] > self.thres], x_test[:,Xnew[j, :] > self.thres], y_train, y_test)

                    # Greedy selection
                    if Fnew < fit[i]:
                        fit[i] = Fnew
                        X[i, :] = Xnew[j, :]

            # Best solution (6)
            fitG_new = np.min(fit)
            idx = np.argmin(fit)
            Xgb_new = X[idx, :]

            # Best update
            if fitG_new < fitG:
                fitG = fitG_new
                Xgb = Xgb_new

            # Store
            curve[t - 1] = fitG
            print('\nIteration %d Best (TSA)= %f' % (t, curve[t - 1]))
            t += 1

        # Select features
        Pos = np.arange(dim)
        Sf = Pos[Xgb > self.thres]

        # Store results
        TSA = {'sf': Sf, 'c': curve}
        return TSA