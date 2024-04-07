import numpy as np

from losses.jFitnessFunction import jFitnessFunction

class jSatinBowerBirdOptimization:
    def __init__(self, N, max_Iter, loss_func, alpha=0.9, beta=0.1, thres=0.5, tau=1, rho=0.2, eta=1, z=0.02, MR=0.05):
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
        self.MR=MR

    def optimize(self, x_train, x_test, y_train, y_test):

        # Parameters
        lb = 0
        ub = 1
        z = self.z    # constant
        MR = self.MR    # mutation rate
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
        fitE = np.inf
        for i in range(N):
            fit[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)
            # Elite update
            if fit[i] < fitE:
                fitE = fit[i]
                Xe = X[i, :]

        # Sigma (7)
        sigma = z * (ub - lb)

        # Pre
        Xnew = np.zeros((N, dim))
        Fnew = np.zeros(N)

        curve = np.zeros(self.max_Iter)
        curve[0] = fitE
        t = 2

        # Iterations
        while t <= self.max_Iter:
            # Calculate probability (1-2)
            Ifit = 1 / (1 + fit)
            prob = Ifit / sum(Ifit)
            for i in range(N):
                for d in range(dim):
                    # Select a bower using roulette wheel
                    rw = self._roulette_wheel_selection(prob)
                    # Compute lambda (4)
                    lambd = self.alpha / (1 + prob[rw])
                    # Update position (3)
                    Xnew[i, d] = X[i, d] + lambd * (((X[rw, d] + Xe[d]) / 2) - X[i, d])
                    # Mutation
                    if np.random.rand() <= MR:
                        # Normal distribution & Position update (5-6)
                        r_normal = np.random.randn()
                        Xnew[i, d] = X[i, d] + (sigma * r_normal)

                # Boundary
                XB = Xnew[i, :]
                XB[XB > ub] = ub
                XB[XB < lb] = lb
                Xnew[i, :] = XB

            # Fitness
            for i in range(N):
                Fnew[i] = self.loss_func(x_train[:, Xnew[i, :] > self.thres], x_test[:, Xnew[i, :] > self.thres],
                                        y_train, y_test)

            # Merge & Select best N solutions
            XX = np.concatenate((X, Xnew), axis=0)
            FF = np.concatenate((fit, Fnew))
            idx = np.argsort(FF)
            X = XX[idx[0:N], :]
            fit = FF[0:N]

            # Elite update
            if fit[0] < fitE:
                fitE = fit[0]
                Xe = X[0, :]

            # Save
            curve[t-1] = fitE
            print('\nIteration %d Best (SBO) = %f' % (t, curve[t-1]))
            t = t + 1

        # Select features
        Pos = np.arange(1, dim+1)
        Sf = Pos[(Xe > self.thres) == 1] - 1

        # Store results
        SBO = {}
        SBO['sf'] = Sf
        SBO['c'] = curve

        return SBO

    @staticmethod
    # Roulette Wheel Selection
    def _roulette_wheel_selection(prob):
        # Cummulative summation
        C = np.cumsum(prob)
        # Random one value, most probability value [0~1]
        P = np.random.rand()
        # Route wheel
        for i in range(len(C)):
            if C[i] > P:
                return i