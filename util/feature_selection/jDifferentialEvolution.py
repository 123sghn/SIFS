import numpy as np

from losses.jFitnessFunction import jFitnessFunction

class jDifferentialEvolution:
    def __init__(self, N, max_Iter, loss_func, alpha=0.9, beta=0.1, thres=0.5, tau=1, rho=0.2, eta=1, CR=0.9, F=0.5):
        self.N = N
        self.max_Iter = max_Iter
        self.loss_func = loss_func
        self.alpha = alpha
        self.beta = beta
        self.thres = thres
        self.tau = tau
        self.rho = rho
        self.eta = eta
        self.CR = CR
        self.F = F

    def optimize(self, x_train, x_test, y_train, y_test):
        # Parameters
        lb = 0
        ub = 1
        CR = self.CR  # crossover rate
        F = self.F  # constant factor
        N = self.N

        # Function
        fun = jFitnessFunction
        # Dimension
        dim = x_train.shape[1]
        # Initialize positions
        X = np.zeros((N, dim))
        for i in range(N):
            for d in range(dim):
                X[i, d] = lb + (ub - lb) * np.random.rand()

        # Fitness
        fit = np.zeros(N)
        fitG = np.inf
        for i in range(N):
            fit[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)
            # Best update
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i, :]

        # Pre
        U = np.zeros((N, dim))
        V = np.zeros((N, dim))

        curve = np.zeros(self.max_Iter)
        curve[0] = fitG
        t = 2
        while t <= self.max_Iter:
            for i in range(N):
                # Choose r1, r2, r3 randomly, but not equal to i & each other
                RN = np.random.permutation(N)
                RN = RN[RN != i]
                r1 = RN[0]
                r2 = RN[1]
                r3 = RN[2]
                # Mutation
                for d in range(dim):
                    V[i, d] = X[r1, d] + F * (X[r2, d] - X[r3, d])
                # Random select a index [1, D]
                rnbr = np.random.randint(1, dim + 1)
                # Crossover
                for d in range(dim):
                    if np.random.rand() <= CR or d == rnbr:
                        U[i, d] = V[i, d]
                    else:
                        U[i, d] = X[i, d]
                # Boundary
                XB = U[i, :]
                XB[XB > ub] = ub
                XB[XB < lb] = lb
                U[i, :] = XB
                # Fitness
                Fnew = self.loss_func(x_train[:, U[i, :] > self.thres],
                                             x_test[:, U[i, :] > self.thres], y_train, y_test)

                # Selection
                if Fnew <= fit[i]:
                    X[i, :] = U[i, :]
                    fit[i] = Fnew
                # Best update
                if fit[i] < fitG:
                    fitG = fit[i]
                    Xgb = X[i, :]

            curve[t - 1] = fitG
            print('\nIteration %d Best (DE)= %f' % (t, fitG))
            t = t + 1

        # Select features based on selected index
        Pos = np.arange(dim)
        Sf = Pos[Xgb > self.thres]

        # Store results
        DE = {}
        DE['sf'] = Sf
        DE['c'] = curve

        return DE