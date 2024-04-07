import numpy as np

from losses.jFitnessFunction import jFitnessFunction

class jHarmonySearch:
    def __init__(self, N, max_Iter, loss_func, alpha=0.9, beta=0.1, thres=0.5, tau=1, rho=0.2, eta=1, PAR=0.05, HMCR=0.7, bw=0.2):
        self.N = N
        self.max_Iter = max_Iter
        self.loss_func = loss_func
        self.alpha = alpha
        self.beta = beta
        self.thres = thres
        self.tau = tau
        self.rho = rho
        self.eta = eta
        self.PAR = PAR
        self.HMCR = HMCR
        self.bw = bw

    def optimize(self, x_train, x_test, y_train, y_test):
        # Parameters
        lb = 0
        ub = 1
        PAR = self.PAR  # pitch adjusting rate
        HMCR = self.HMCR  # harmony memory considering rate
        bw = self.bw  # bandwidth
        HMS = self.N

        # Number of dimensions
        dim = x_train.shape[1]
        # Initial (13)
        X = np.zeros((HMS, dim))
        for i in range(HMS):
            for d in range(dim):
                X[i, d] = lb + (ub - lb) * np.random.rand()

        # Fitness
        fit = np.zeros(HMS)
        fitG = np.inf
        for i in range(HMS):
            fit[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)
            # Best update
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i, :]

        # Worst solution
        fitW = np.max(fit)
        idx_W = np.argmax(fit)

        # Pre
        Xnew = np.zeros((HMS, dim))

        curve = np.zeros(self.max_Iter)
        curve[0] = fitG
        t = 2
        while t <= self.max_Iter:
            for i in range(HMS):
                for d in range(dim):
                    # Harmony memory considering rate
                    if np.random.rand() < HMCR:
                        # Random select 1 harmony memory
                        k = np.random.randint(1, HMS + 1)
                        # Update new harmony using harmony memory
                        Xnew[i, d] = X[k - 1, d]
                    else:
                        # Randomize a new harmony
                        Xnew[i, d] = lb + (ub - lb) * np.random.rand()

                    # Pitch adjusting rate
                    if np.random.rand() < PAR:
                        r = np.random.rand()
                        if r > 0.5:
                            Xnew[i, d] = X[i, d] + np.random.rand() * bw
                        else:
                            Xnew[i, d] = X[i, d] - np.random.rand() * bw

                # Boundary
                XB = Xnew[i, :]
                XB[XB > ub] = ub
                XB[XB < lb] = lb
                Xnew[i, :] = XB

            # Fitness
            for i in range(HMS):
                # Fitness
                Fnew = self.loss_func(x_train[:, Xnew[i, :] > self.thres], x_test[:, Xnew[i, :] > self.thres],
                                        y_train, y_test)
                # Update worst solution
                if Fnew < fitW:
                    fit[idx_W] = Fnew
                    X[idx_W, :] = Xnew[i, :]
                    # New worst solution
                    fitW = np.max(fit)
                    idx_W = np.argmax(fit)

                # Global update
                if Fnew < fitG:
                    fitG = Fnew
                    Xgb = Xnew[i, :]

            curve[t - 1] = fitG
            print('\nIteration %d Best (HS)= %f' % (t, curve[t - 1]))
            t = t + 1

        # Select features
        Pos = np.arange(dim)
        Sf = Pos[Xgb > self.thres]

        # Store results
        HS = {}
        HS['sf'] = Sf
        HS['c'] = curve

        return HS