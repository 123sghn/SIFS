import numpy as np

from losses.jFitnessFunction import jFitnessFunction

class jPathFinderAlgorithm:
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

        # Fitness
        fit = np.zeros(N)
        fitP = np.inf
        for i in range(N):
            fit[i] = self.loss_func(x_train[:,X[i, :] > self.thres], x_test[:,X[i, :] > self.thres], y_train, y_test)

            # Pathfinder update
            if fit[i] < fitP:
                fitP = fit[i]
                Xpf = X[i, :]

        # Set previous pathfiner
        Xpf_old = Xpf
        # Pre
        Xpf_new = np.zeros(dim)
        Xnew = np.zeros((N, dim))

        curve = np.zeros(self.max_Iter)
        curve[0] = fitP
        t = 1
        # Iterations
        while t < self.max_Iter:
            # Alpha & beta in [1,2]
            alpha = 1 + np.random.rand()
            beta = 1 + np.random.rand()
            for d in range(dim):
                # Define u2 in [-1,1]
                u2 = -1 + 2 * np.random.rand()
                # Compute A (2.6)
                A = u2 * np.exp(-(2 * t) / self.max_Iter)
                # Update pathfinder (2.4)
                r3 = np.random.rand()
                Xpf_new[d] = Xpf[d] + 2 * r3 * (Xpf[d] - Xpf_old[d]) + A
            # Boundary
            Xpf_new[Xpf_new > ub] = ub
            Xpf_new[Xpf_new < lb] = lb
            # Update previous path
            Xpf_old = Xpf
            # Fitness
            Fnew = self.loss_func(x_train[:, Xpf_new > self.thres], x_test[:, Xpf_new > self.thres], y_train, y_test)

            # Greedy selection
            if Fnew < fitP:
                fitP = Fnew
                Xpf = Xpf_new
            # Sort member
            idx = np.argsort(fit)
            fit = fit[idx]
            X = X[idx, :]
            # Update first solution
            if Fnew < fit[0]:
                fit[0] = Fnew
                X[0, :] = Xpf_new
            # Update
            for i in range(1, N):
                # Distance (2.5)
                Dij = np.linalg.norm(X[i, :] - X[i - 1, :])
                for d in range(dim):
                    # Define u1 in [-1,1]
                    u1 = -1 + 2 * np.random.rand()
                    # Compute epsilon (2.5)
                    eps = (1 - (t / self.max_Iter)) * u1 * Dij
                    # Define R1, R2
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    R1 = alpha * r1
                    R2 = beta * r2
                    # Update member (2.3)
                    Xnew[i, d] = X[i, d] + R1 * (X[i - 1, d] - X[i, d]) + R2 * (Xpf[d] - X[i, d]) + eps
                # Boundary
                XB = Xnew[i, :]
                XB[XB > ub] = ub
                XB[XB < lb] = lb
                Xnew[i, :] = XB
            # Fitness
            for i in range(1, N):
                # Fitness
                Fnew = self.loss_func(x_train[:,Xnew[i, :] > self.thres], x_test[:,Xnew[i, :] > self.thres], y_train, y_test)

                # Selection
                if Fnew < fit[i]:
                    fit[i] = Fnew
                    X[i, :] = Xnew[i, :]
                # Pathfinder update
                if fit[i] < fitP:
                    fitP = fit[i]
                    Xpf = X[i, :]
            curve[t - 1] = fitP
            print('\nIteration %d Best (PFA)= %f' % (t, curve[t]))
            t = t + 1

        # Select features
        Pos = np.arange(dim)
        Sf = Pos[Xpf > self.thres]

        # Store results
        PFA = {}
        PFA['sf'] = Sf
        PFA['c'] = curve

        return PFA
