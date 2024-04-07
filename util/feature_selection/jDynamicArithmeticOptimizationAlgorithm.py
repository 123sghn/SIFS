import numpy as np
from losses.jFitnessFunction import jFitnessFunction

class jDynamicArithmeticOptimizationAlgorithm:
    def __init__(self, N, max_Iter, loss_func, alpha=0.9, beta=0.1, thres=0.5, tau=1, rho=0.2, eta=1, Mu=0.001):
        self.N = N
        self.max_Iter = max_Iter
        self.loss_func = loss_func
        self.alpha = alpha
        self.beta = beta
        self.thres = thres
        self.tau = tau
        self.rho = rho
        self.eta = eta
        self.Mu = Mu

    def optimize(self, x_train, x_test, y_train, y_test):

        dim = x_train.shape[1]  # Number of Variable
        lb = np.zeros(dim)  # Upper Bound
        ub = np.ones(dim)  # Lower Bound
        Mu = self.Mu
        N = self.N

        Xgb = np.zeros(dim)
        fitG = np.inf
        curve = np.zeros(self.max_Iter)

        # Initialize the positions of solution
        X = np.random.rand(N, dim) * (ub - lb) + lb
        Xnew = X.copy()
        Ffun = np.zeros(X.shape[0])  # (fitness values)
        Ffun_new = np.zeros(Xnew.shape[0])  # (fitness values)

        t = 1

        for i in range(X.shape[0]):
            Ffun[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)  # Calculate the fitness values of solutions
            if Ffun[i] < fitG:
                fitG = Ffun[i]
                Xgb = X[i, :]

        # Main Loop
        while t < self.max_Iter + 1:  # Main loop
            DAF = (self.max_Iter + 1 / t) ** (self.alpha)  # DAF2

            DCS = 0.99 * ((1 - (t / self.max_Iter) ** 0.5))  # DCS

            # Upate the Position of solutions
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    r1 = np.random.rand()
                    if len(lb) == 1:
                        if r1 < DAF:
                            r2 = np.random.rand()
                            if r2 > 0.5:
                                Xnew[i, j] = (Xgb[j] / (DCS + 1e-7) * ((ub - lb) * Mu + lb))
                            else:
                                Xnew[i, j] = (Xgb[j] * DCS * ((ub - lb) * Mu + lb))
                        else:
                            r3 = np.random.rand()
                            if r3 > 0.5:
                                Xnew[i, j] = (Xgb[j] - DCS * ((ub - lb) * Mu + lb))
                            else:
                                Xnew[i, j] = (Xgb[j] + DCS * ((ub - lb) * Mu + lb))
                    elif len(lb) != 1:
                        r1 = np.random.rand()
                        if r1 < DAF:
                            r2 = np.random.rand()
                            if r2 > 0.5:
                                Xnew[i, j] = (Xgb[j] / (DCS + 1e-7) * ((ub[j] - lb[j]) * Mu + lb[j]))
                            else:
                                Xnew[i, j] = (Xgb[j] * DCS * ((ub[j] - lb[j]) * Mu + lb[j]))
                        else:
                            r3 = np.random.rand()
                            if r3 > 0.5:
                                Xnew[i, j] = (Xgb[j] - DCS * ((ub[j] - lb[j]) * Mu + lb[j]))
                            else:
                                Xnew[i, j] = (Xgb[j] + DCS * ((ub[j] - lb[j]) * Mu + lb[j]))

                Flag_UB = Xnew[i, :] > ub  # check if they exceed (up) the boundaries
                Flag_LB = Xnew[i, :] < lb  # check if they exceed (down) the boundaries
                Xnew[i, :] = (Xnew[i, :] * ~(Flag_UB + Flag_LB)) + ub * Flag_UB + lb * Flag_LB

                Ffun_new[i] = self.loss_func(x_train[:, Xnew[i, :] > self.thres], x_test[:, Xnew[i, :] > self.thres],
                                        y_train, y_test)  # calculate Fitness function
                if Ffun_new[i] < Ffun[i]:
                    X[i, :] = Xnew[i, :]
                    Ffun[i] = Ffun_new[i]
                if Ffun[i] < fitG:
                    fitG = Ffun[i]
                    Xgb = X[i, :]

            # Update the convergence curve
            curve[t - 1] = fitG
            print('\nIteration {} Best (DAOA)= {}'.format(t, curve[t - 1]))
            t += 1  # incremental iteration

        Pos = np.arange(1, dim + 1)
        Sf = Pos[(Xgb > self.thres) == 1] - 1
        
        DAOA = {}
        DAOA['sf'] = Sf  # best_X
        DAOA['c'] = curve

        return DAOA