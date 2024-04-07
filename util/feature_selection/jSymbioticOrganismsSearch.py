import numpy as np

from losses.jFitnessFunction import jFitnessFunction

class jSymbioticOrganismsSearch:
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

        # Objective function
        fun = jFitnessFunction
        # Number of dimensions
        dim = x_train.shape[1]
        # Initial
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
            # Global best
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i, :]
        # Pre
        Xi = np.zeros(dim)
        Xj = np.zeros(dim)

        curve = np.zeros(self.max_Iter)
        curve[0] = fitG
        t = 2
        # Iteration
        while t <= self.max_Iter:
            for i in range(N):
                # {1} Mutualism phase
                R = np.random.permutation(N)
                R = R[R != i]
                J = R[0]
                # Benefit factor [1 or 2]
                BF1 = np.random.randint(1, 3)
                BF2 = np.random.randint(1, 3)
                for d in range(dim):
                    # Mutual vector (3)
                    MV = (X[i, d] + X[J, d]) / 2
                    # Update solution (1-2)
                    Xi[d] = X[i, d] + np.random.rand() * (Xgb[d] - MV * BF1)
                    Xj[d] = X[J, d] + np.random.rand() * (Xgb[d] - MV * BF2)
                # Boundary
                Xi[Xi > ub] = ub
                Xi[Xi < lb] = lb
                Xj[Xj > ub] = ub
                Xj[Xj < lb] = lb
                # Fitness
                fitI = self.loss_func(x_train[:, Xi > self.thres], x_test[:, Xi > self.thres],
                                        y_train, y_test)
                fitJ = self.loss_func(x_train[:, Xj > self.thres], x_test[:, Xj > self.thres],
                                        y_train, y_test)
                # Update if better solution
                if fitI < fit[i]:
                    fit[i] = fitI
                    X[i, :] = Xi
                if fitJ < fit[J]:
                    fit[J] = fitJ
                    X[J, :] = Xj

                # {2} Commensalism phase
                R = np.random.permutation(N)
                R = R[R != i]
                J = R[0]
                for d in range(dim):
                    # Random number in [-1,1]
                    r1 = -1 + 2 * np.random.rand()
                    # Update solution (4)
                    Xi[d] = X[i, d] + r1 * (Xgb[d] - X[J, d])
                # Boundary
                Xi[Xi > ub] = ub
                Xi[Xi < lb] = lb
                # Fitness
                fitI = self.loss_func(x_train[:, Xi > self.thres], x_test[:, Xi > self.thres],
                                        y_train, y_test)
                # Update if better solution
                if fitI < fit[i]:
                    fit[i] = fitI
                    X[i, :] = Xi

                # {3} Parasitism phase
                R = np.random.permutation(N)
                R = R[R != i]
                J = R[0]
                # Parasite vector
                PV = X[i, :].copy()
                # Randomly select random variables
                r_dim = np.random.permutation(dim)
                dim_no = np.random.randint(1, dim + 1)
                for d in range(dim_no):
                    # Update solution
                    PV[r_dim[d]] = lb + (ub - lb) * np.random.rand()
                # Boundary
                PV[PV > ub] = ub
                PV[PV < lb] = lb
                # Fitness
                fitPV = self.loss_func(x_train[:, PV > self.thres], x_test[:, PV > self.thres],
                                        y_train, y_test)
                # Replace parasite if it is better than j
                if fitPV < fit[J]:
                    fit[J] = fitPV
                    X[J, :] = PV

            # Update global best
            for i in range(N):
                if fit[i] < fitG:
                    fitG = fit[i]
                    Xgb = X[i, :]

            curve[t-1] = fitG
            print('\nIteration', t, 'GBest (SOS)=', curve[t-1])
            t = t + 1

        # Select features based on selected index
        Pos = np.arange(dim)
        Sf = Pos[Xgb > self.thres]
        # Store results
        SOS = {}
        SOS['sf'] = Sf
        SOS['c'] = curve

        return SOS
