import numpy as np

from losses.jFitnessFunction import jFitnessFunction

class jMantaRayForagingOptimization:
    def __init__(self, N, max_Iter, loss_func, alpha=0.9, beta=0.1, thres=0.5, tau=1, rho=0.2, eta=1, S=2):
        self.N = N
        self.max_Iter = max_Iter
        self.loss_func = loss_func
        self.alpha = alpha
        self.beta = beta
        self.thres = thres
        self.tau = tau
        self.rho = rho
        self.eta = eta
        self.S = S

    def optimize(self, x_train, x_test, y_train, y_test):

        # Parameters
        lb = 0
        ub = 1
        S = self.S  # somersault factor
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
        fitG = float('inf')
        for i in range(N):
            fit[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)
            # Best solution
            if fit[i] < fitG:
                fitG = fit[i]
                Xbest = X[i, :].copy()
        # Pre
        Xnew = np.zeros((N, dim))

        curve = np.zeros(self.max_Iter)
        curve[0] = fitG
        t = 1
        # Iteration
        while t < self.max_Iter:
            for i in range(N):
                # [Cyclone foraging]
                if np.random.rand() < 0.5:
                    if t / self.max_Iter < np.random.rand():
                        # Compute beta (5)
                        r1 = np.random.rand()
                        beta = 2 * np.exp(r1 * ((self.max_Iter - t + 1) / self.max_Iter)) * \
                            (np.sin(2 * np.pi * r1))
                        for d in range(dim):
                            # Create random solution (6)
                            Xrand = lb + np.random.rand() * (ub - lb)
                            # First manta ray follow best food (7)
                            if i == 0:
                                Xnew[i, d] = Xrand + np.random.rand() * (Xrand - X[i, d]) + \
                                    beta * (Xrand - X[i, d])
                            # Followers follew the front manta ray (7)
                            else:
                                Xnew[i, d] = Xrand + np.random.rand() * (X[i - 1, d] - X[i, d]) + \
                                    beta * (Xrand - X[i, d])
                    else:
                        # Compute beta (5)
                        r1 = np.random.rand()
                        beta = 2 * np.exp(r1 * ((self.max_Iter - t + 1) / self.max_Iter)) * \
                            (np.sin(2 * np.pi * r1))
                        for d in range(dim):
                            # First manta ray follow best food (4)
                            if i == 0:
                                Xnew[i, d] = Xbest[d] + np.random.rand() * (Xbest[d] - X[i, d]) + \
                                    beta * (Xbest[d] - X[i, d])
                            # Followers follow the front manta ray (4)
                            else:
                                Xnew[i, d] = Xbest[d] + np.random.rand() * (X[i - 1, d] - X[i, d]) + \
                                    beta * (Xbest[d] - X[i, d])
                # [Chain foraging]
                else:
                    for d in range(dim):
                        # Compute alpha (2)
                        r = np.random.rand()
                        alpha = 2 * r * np.sqrt(abs(np.log(r)))
                        # First manta ray follow best food (1)
                        if i == 0:
                            Xnew[i, d] = X[i, d] + np.random.rand() * (Xbest[d] - X[i, d]) + \
                                alpha * (Xbest[d] - X[i, d])
                        # Followers follew the front manta ray (1)
                        else:
                            Xnew[i, d] = X[i, d] + np.random.rand() * (X[i - 1, d] - X[i, d]) + \
                                alpha * (Xbest[d] - X[i, d])
                # Boundary
                XB = Xnew[i, :].copy()
                XB[XB > ub] = ub
                XB[XB < lb] = lb
                Xnew[i, :] = XB
            # Fitness
            for i in range(N):
                Fnew = self.loss_func(x_train[:, Xnew[i, :] > self.thres], x_test[:, Xnew[i, :] > self.thres],
                                        y_train, y_test)
                # Greedy selection
                if Fnew < fit[i]:
                    fit[i] = Fnew
                    X[i, :] = Xnew[i, :].copy()
                # Update best
                if fit[i] < fitG:
                    fitG = fit[i]
                    Xbest = X[i, :].copy()
            # [Somersault foraging]
            for i in range(N):
                # Manta ray update (8)
                r2 = np.random.rand()
                r3 = np.random.rand()
                for d in range(dim):
                    Xnew[i, d] = X[i, d] + S * (r2 * Xbest[d] - r3 * X[i, d])
                # Boundary
                XB = Xnew[i, :].copy()
                XB[XB > ub] = ub
                XB[XB < lb] = lb
                Xnew[i, :] = XB
            # Fitness
            for i in range(N):
                Fnew = self.loss_func(x_train[:, Xnew[i, :] > self.thres], x_test[:, Xnew[i, :] > self.thres],
                                        y_train, y_test)
                # Greedy selection
                if Fnew < fit[i]:
                    fit[i] = Fnew
                    X[i, :] = Xnew[i, :].copy()
                # Update best
                if fit[i] < fitG:
                    fitG = fit[i]
                    Xbest = X[i, :].copy()
            curve[t-1] = fitG
            print('\nIteration %d Best (MRFO)= %f' % (t, curve[t-1]))
            t = t + 1
        # Select features based on selected index
        Pos = np.arange(dim)
        Sf = Pos[Xbest > self.thres]
        # Store results
        MRFO = {}
        MRFO['sf'] = Sf
        MRFO['c'] = curve

        return MRFO
