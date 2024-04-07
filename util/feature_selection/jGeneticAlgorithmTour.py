import numpy as np

from losses.jFitnessFunction import jFitnessFunction

class jGeneticAlgorithmTour:
    def __init__(self, N, max_Iter, loss_func, alpha=0.9, beta=0.1, thres=0.5, tau=1, rho=0.2, eta=1, CR=0.8, MR=0.01, Tour_size=3):
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
        self.MR = MR
        self.Tour_size = Tour_size

    def optimize(self, x_train, x_test, y_train, y_test):

        # Parameters
        CR = self.CR  # crossover rate
        MR = self.MR  # mutation rate
        Tour_size = self.Tour_size  # tournament size
        N = self.N

        # Number of dimensions
        dim = x_train.shape[1]

        # Initial
        X = self._Initialization(N, dim)

        # Fitness
        fit = np.zeros(N)
        fitG = np.inf
        for i in range(N):
            fit[i] = self.loss_func(x_train[:,X[i, :] > self.thres], x_test[:,X[i, :] > self.thres], y_train, y_test)

            # Best update
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i, :]

        # Pre
        curve = np.zeros(self.max_Iter)
        curve[0] = fitG
        t = 2

        # Generations
        while t <= self.max_Iter:
            # Preparation
            Xc1 = np.zeros((N, dim))
            Xc2 = np.zeros((N, dim))
            fitC1 = np.ones(N)
            fitC2 = np.ones(N)
            z = 1
            for i in range(N):
                if np.random.rand() < CR:
                    # Select two parents
                    k1 = self._Tournament_Selection(fit, Tour_size, N)
                    k2 = self._Tournament_Selection(fit, Tour_size, N)
                    # Store parents
                    P1 = X[k1, :]
                    P2 = X[k2, :]
                    # Single point crossover
                    ind = np.random.randint(1, dim - 1)
                    # Crossover between two parents
                    Xc1[z] = np.concatenate((P1[:ind+1], P2[ind + 1:dim]))
                    Xc2[z] = np.concatenate((P2[:ind+1], P1[ind + 1:dim]))
                    # Mutation
                    for d in range(dim):
                        # First child
                        if np.random.rand() < MR:
                            Xc1[z, d] = 1 - Xc1[z, d]
                        # Second child
                        if np.random.rand() < MR:
                            Xc2[z, d] = 1 - Xc2[z, d]
                    # Fitness
                    fitC1[z] = self.loss_func(x_train[:,Xc1[i, :] > self.thres], x_test[:,Xc1[i, :] > self.thres], y_train, y_test)
                    fitC2[z] = self.loss_func(x_train[:,Xc2[i, :] > self.thres], x_test[:,Xc2[i, :] > self.thres], y_train, y_test)
                    z = z + 1
            # Merge population
            XX = np.concatenate((X, Xc1, Xc2), axis=0)
            FF = np.concatenate((fit, fitC1, fitC2), axis=0)
            # Select N best solution
            idx = np.argsort(FF)
            X = XX[idx[:N], :]
            fit = FF[:N]
            # Best agent
            if fit[0] < fitG:
                fitG = fit[0]
                Xgb = X[0, :]
            # Save
            curve[t - 1] = fitG
            print('\nGeneration %d Best (GA Tournament)= %f' % (t, curve[t - 1]))
            t = t + 1

        # Select features based on selected index
        Pos = np.arange(dim)
        Sf = Pos[Xgb == 1]

        # Store results
        GA = {}
        GA['sf'] = Sf
        GA['c'] = curve

        return GA


    @staticmethod
    def _Tournament_Selection(fit, Tour_size, N):

        # Random positions based on position & Tournament Size
        Tour_idx = np.random.choice(N, Tour_size)
        # Select fitness value based on position selected by tournament
        Tour_fit = fit[Tour_idx]
        # Get position of best fitness value (win tournament)
        idx = np.argmin(Tour_fit)
        # Store the position
        Index = Tour_idx[idx]

        return Index

    @staticmethod
    def _Initialization(N, dim):
        # Initialize X vectors
        X = np.zeros((N, dim))
        for i in range(N):
            for d in range(dim):
                if np.random.rand() > 0.5:
                    X[i][d] = 1

        return X
