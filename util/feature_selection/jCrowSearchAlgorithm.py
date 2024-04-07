import numpy as np

from losses.jFitnessFunction import jFitnessFunction


class jCrowSearchAlgorithm:
    def __init__(
        self,
        N,
        max_Iter,
        loss_func,
        alpha=0.9,
        beta=0.1,
        thres=0.5,
        tau=1,
        rho=0.2,
        eta=1,
        AP=0.1,
        f1=1.5,
    ):
        self.loss_func = loss_func
        self.tau = tau
        self.eta = eta
        self.max_Iter = max_Iter
        self.N = N
        self.thres = thres
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.AP = AP
        self.f1 = f1

    def optimize(self, x_train, x_test, y_train, y_test):
        # Parameters
        lb = 0
        ub = 1
        AP = self.AP
        fl = self.f1
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
        fitG = np.inf
        for i in range(N):
            fit[i] = self.loss_func(
                x_train[:, X[i, :] > self.thres],
                x_test[:, X[i, :] > self.thres],
                y_train,
                y_test,
            )
            # Global update
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i, :]
        # Save memory
        fitM = fit
        Xm = X
        # Pre
        Xnew = np.zeros((N, dim))

        curve = np.zeros(self.max_Iter)
        curve[0] = fitG
        t = 2
        # Iteration
        while t <= self.max_Iter:
            for i in range(N):
                # Random select 1 memory crow to follow
                k = np.random.randint(1, N + 1)
                # Awareness of crow m (2)
                if np.random.rand() >= AP:
                    r = np.random.rand()
                    for d in range(dim):
                        # Crow m does not know it has been followed (1)
                        Xnew[i, d] = X[i, d] + r * fl * (Xm[k - 1, d] - X[i, d])
                else:
                    for d in range(dim):
                        # Crow m fools crow i by flying randomly
                        Xnew[i, d] = lb + (ub - lb) * np.random.rand()
            # Fitness
            for i in range(N):
                # Fitness
                Fnew = self.loss_func(
                    x_train[:, Xnew[i, :] > self.thres],
                    x_test[:, Xnew[i, :] > self.thres],
                    y_train,
                    y_test,
                )
                # Check feasibility
                if np.all(Xnew[i, :] >= lb) and np.all(Xnew[i, :] <= ub):
                    # Update crow
                    X[i, :] = Xnew[i, :]
                    fit[i] = Fnew
                    # Memory update (5)
                    if fit[i] < fitM[i]:
                        Xm[i, :] = X[i, :]
                        fitM[i] = fit[i]
                    # Global update
                    if fitM[i] < fitG:
                        fitG = fitM[i]
                        Xgb = Xm[i, :]
            curve[t - 1] = fitG
            print("\nIteration", t, "Best (CSA)=", curve[t - 1])
            t = t + 1
        # Select features
        Pos = np.arange(dim)
        Sf = Pos[(Xgb > self.thres) == 1]
        # Store results
        CSA = {}
        CSA["sf"] = Sf
        CSA["c"] = curve
        return CSA
