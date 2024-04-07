import numpy as np

from losses.jFitnessFunction import jFitnessFunction

class jButterflyOptimizationAlgorithm:
    def __init__(self, N, max_Iter, loss_func, alpha=0.9, beta=0.1, thres=0.5, tau=1, rho=0.2, eta=1, c=0.01, p=0.8):
        self.loss_func = loss_func
        self.tau = tau
        self.eta = eta
        self.max_Iter = max_Iter
        self.N = N
        self.thres = thres
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.c = c
        self.p = p

    def optimize(self, x_train, x_test, y_train, y_test):
        # Parameters
        lb = 0
        ub = 1
        c = self.c  # modular modality
        p = self.p  # switch probability
        N = self.N

        # Number of dimensions
        dim = x_train.shape[1]
        # Initial
        X = np.zeros((N, dim))
        for i in range(N):
            for d in range(dim):
                X[i, d] = lb + (ub - lb) * np.random.rand()
        # Pre
        Xnew = np.zeros((N, dim))
        fitG = np.inf
        fit = np.zeros(N)

        curve = np.zeros(self.max_Iter)
        t = 1
        # Iterations
        while t <= self.max_Iter:
            # Fitness
            for i in range(N):
                fit[i] = self.loss_func(x_train[:,X[i, :] > self.thres], x_test[:,X[i, :] > self.thres], y_train, y_test)
                # Global update
                if fit[i] < fitG:
                    fitG = fit[i]
                    Xgb = X[i, :]
            # Power component, increase from 0.1 to 0.3
            a = 0.1 + 0.2 * (t / self.max_Iter)
            for i in range(N):
                # Compute fragrance (1)
                f = c * (fit[i] ** a)
                # Random number in [0,1]
                r = np.random.rand()
                if r < p:
                    r1 = np.random.rand()
                    for d in range(dim):
                        # Move toward best butterfly (2)
                        Xnew[i, d] = X[i, d] + ((r1 ** 2) * Xgb[d] - X[i, d]) * f
                else:
                    # Random select two butterfly
                    R = np.random.permutation(N)
                    J = R[0]
                    K = R[1]
                    r2 = np.random.rand()
                    for d in range(dim):
                        # Move randomly (3)
                        Xnew[i, d] = X[i, d] + ((r2 ** 2) * X[J, d] - X[K, d]) * f
                # Boundary
                XB = Xnew[i, :].copy()
                XB[XB > ub] = ub
                XB[XB < lb] = lb
                Xnew[i, :] = XB
            # Replace
            X = Xnew
            # Save
            curve[t - 1] = fitG
            print('\nIteration %d Best (BOA)= %f' % (t, curve[t - 1]))
            t = t + 1
        # Select features
        Pos = np.arange(dim)
        Sf = Pos[(Xgb > self.thres) == 1]
        # Store results
        BOA = {}
        BOA['sf'] = Sf
        BOA['c'] = curve

        return BOA