import numpy as np

from losses.jFitnessFunction import jFitnessFunction

class jSimulatedAnnealing:
    def __init__(self, N, max_Iter, loss_func, alpha=0.9, beta=0.1, thres=0.5, tau=1, rho=0.2, eta=1, c=0.93, T0=100):
        self.N = N
        self.max_Iter = max_Iter
        self.loss_func = loss_func
        self.alpha = alpha
        self.beta = beta
        self.thres = thres
        self.tau = tau
        self.rho = rho
        self.eta = eta
        self.c = c
        self.T0 = T0

    def optimize(self, x_train, x_test, y_train, y_test):

        # Parameters
        c = self.c  # cooling rate
        T0 = self.T0   # initial temperature

        # Number of dimensions
        dim = x_train.shape[1]
        # Initial
        X = self._Initialization(1, dim)
        X = X.flatten()
        # Fitness
        fit = self.loss_func(x_train[:, X > self.thres], x_test[:, X > self.thres],
                                        y_train, y_test)
        # Initial best
        Xgb = X
        fitG = fit
        # Pre
        curve = np.zeros(self.max_Iter)
        t = 1
        # Iterations
        while t <= self.max_Iter:
            # Probabilty of swap, insert, flip & eliminate
            prob = np.random.randint(1, 5)
            # Swap operation
            if prob == 1:
                Xnew = X.copy()
                # Find index with bit '0' & '1'
                bit0 = np.where(X == 0)[0]
                bit1 = np.where(X == 1)[0]
                len_0 = len(bit0)
                len_1 = len(bit1)
                # Solve issue with missing bit '0' or '1'
                if len_0 != 0 and len_1 != 0:
                    # Get one random index from x1 & x2
                    ind0 = np.random.randint(0, len_0)
                    ind1 = np.random.randint(0, len_1)
                    # Swap between two index
                    Xnew[bit0[ind0]] = 1
                    Xnew[bit1[ind1]] = 0

            # Insert operation
            elif prob == 2:
                Xnew = X.copy()
                # Find index with zero
                bit0 = np.where(X == 0)[0]
                len_0 = len(bit0)
                # Solve problem when all index are '1'
                if len_0 != 0:
                    ind = np.random.randint(0, len_0)
                    # Add one feature
                    Xnew[bit0[ind]] = 1

            # Eliminate operation
            elif prob == 3:
                Xnew = X.copy()
                # Find index with one
                bit1 = np.where(X == 1)[0]
                len_1 = len(bit1)
                # Solve problem when all index are '0'
                if len_1 != 0:
                    ind = np.random.randint(0, len_1)
                    # Remove one feature
                    Xnew[bit1[ind]] = 0

            # Flip operation
            elif prob == 4:
                Xnew = X.copy()
                # Flip all variables
                Xnew = 1 - Xnew

            # Fitness
            Fnew = self.loss_func(x_train[:, Xnew > self.thres], x_test[:, Xnew > self.thres],
                                        y_train, y_test)
            # Global best update
            if Fnew <= fitG:
                Xgb = Xnew
                fitG = Fnew
                X = Xnew
            # Accept worst solution with probability
            else:
                # Delta energy
                delta = Fnew - fitG
                # Boltzmann Probility
                P = np.exp(-delta / T0)
                if np.random.random() <= P:
                    X = Xnew
            # Temperature update
            T0 = c * T0
            # Save
            curve[t-1] = fitG
            print('\nIteration {} Best (SA) = {}'.format(t, curve[t-1]))
            t = t + 1

        # Select features
        Pos = np.arange(dim)
        Sf = Pos[Xgb == 1]
        # Store results
        SA = {}
        SA['sf'] = Sf.tolist()
        SA['c'] = curve.tolist()
        return SA

    @staticmethod
    def _Initialization(N, dim):
        # Initialize X vectors
        X = np.zeros((N, dim))
        for i in range(N):
            for d in range(dim):
                if np.random.random() > 0.5:
                    X[i, d] = 1
        return X

