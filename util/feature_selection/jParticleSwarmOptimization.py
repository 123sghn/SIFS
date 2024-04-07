import numpy as np

from losses.jFitnessFunction import jFitnessFunction

class jParticleSwarmOptimization:
    def __init__(self, N, max_Iter, loss_func, alpha=0.9, beta=0.1, thres=0.5, tau=1, rho=0.2, eta=1,c1=2,c2=2,w=0.9):
        self.N = N
        self.max_Iter = max_Iter
        self.loss_func = loss_func
        self.alpha = alpha
        self.beta = beta
        self.thres = thres
        self.tau = tau
        self.rho = rho
        self.eta = eta
        self.c1 = c1
        self.c2 = c2
        self.w = w

    def optimize(self, x_train, x_test, y_train, y_test):

        lb = 0
        ub = 1
        c1 = self.c1
        c2 = self.c2
        w = self.w
        Vmax = (ub - lb) / 2
        N = self.N

        dim = x_train.shape[1]

        X = np.zeros((N, dim))
        V = np.zeros((N, dim))

        for i in range(N):
            for d in range(dim):
                X[i, d] = lb + (ub - lb) * np.random.rand()

        fit = np.zeros(N)
        fitG = np.inf

        for i in range(N):
            fit[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)

            if fit[i] < fitG:
                Xgb = X[i, :]
                fitG = fit[i]

        Xpb = X.copy()
        fitP = fit.copy()

        curve = np.zeros(self.max_Iter)
        curve[0] = fitG
        t = 2

        while t <= self.max_Iter:
            for i in range(N):
                for d in range(dim):
                    r1 = np.random.rand()
                    r2 = np.random.rand()

                    VB = w * V[i, d] + c1 * r1 * (Xpb[i, d] - X[i, d]) + c2 * r2 * (Xgb[d] - X[i, d])
                    VB = np.clip(VB, -Vmax, Vmax)
                    V[i, d] = VB

                    X[i, d] = X[i, d] + V[i, d]

                XB = X[i, :]
                XB = np.clip(XB, lb, ub)
                X[i, :] = XB

                fit[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)

                if fit[i] < fitP[i]:
                    Xpb[i, :] = X[i, :]
                    fitP[i] = fit[i]

                if fitP[i] < fitG:
                    Xgb = Xpb[i, :]
                    fitG = fitP[i]

            curve[t - 1] = fitG
            print('\nIteration {} Best (PSO)= {}'.format(t, curve[t - 1]))
            t += 1

        Pos = np.arange(dim)
        Sf = Pos[Xgb > self.thres]

        PSO = {}
        PSO['sf'] = Sf
        PSO['c'] = curve

        return PSO
