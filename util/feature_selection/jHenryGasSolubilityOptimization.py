import math
import random
import numpy as np

from losses.jFitnessFunction import jFitnessFunction

class jHenryGasSolubilityOptimization:
    def __init__(self, N, max_Iter, loss_func, alpha=0.9, beta=0.1, thres=0.5, tau=1, rho=0.2, eta=1, num_gas=2, K=1, L1=0.005, L2=100,L3=0.01, c1=0.1, c2=0.2):
        self.N = N
        self.max_Iter = max_Iter
        self.loss_func = loss_func
        self.alpha = alpha
        self.beta = beta
        self.thres = thres
        self.tau = tau
        self.rho = rho
        self.eta = eta
        self.num_gas = num_gas
        self.K= K
        self.L1 = L1
        self.L2=L2
        self.L3=L3
        self.c1=c1
        self.c2=c2

    def optimize(self, x_train, x_test, y_train, y_test):

        # Parameters
        lb = 0
        ub = 1
        num_gas = self.num_gas  # number of gas types / cluster
        K = self.K  # constant
        L1 = self.L1
        L2 = self.L2
        L3 = self.L3
        Ttheta = 298.15
        eps = 0.05
        c1 = self.c1
        c2 = self.c2
        N = self.N

        # Number of dimensions
        dim = x_train.shape[1]
        # Number of gas in Nc cluster
        Nn = math.ceil(N / num_gas)
        # Initial (6)
        X = np.zeros((N, dim))
        for i in range(N):
            for d in range(dim):
                X[i, d] = lb + (ub - lb) * random.random()
        # Henry constant & E/R constant (7)
        H = np.zeros(num_gas)
        C = np.zeros(num_gas)
        P = np.zeros((num_gas, Nn))
        for j in range(num_gas):
            H[j] = L1 * random.random()
            C[j] = L3 * random.random()
            for i in range(Nn):
                # Partial pressure (7)
                P[j, i] = L2 * random.random()
        # Divide the population into Nc type of gas cluster
        Cx = []
        for j in range(num_gas):
            if j != num_gas:
                Cx.append(X[(j - 1) * Nn: j * Nn, :])
            else:
                Cx.append(X[(num_gas - 1) * Nn:, :])
        # Fitness of each cluster
        Cfit = []
        fitCB = np.ones(num_gas)
        Cxb = np.zeros((num_gas, dim))
        fitG = float('inf')
        for j in range(num_gas):
            Cfit.append(np.zeros(Cx[j].shape[0]))
            for i in range(Cx[j].shape[0]):
                Cfit[j][i] = self.loss_func(x_train[:, Cx[j][i, :] > self.thres], x_test[:, Cx[j][i, :] > self.thres], y_train, y_test)

                # Update best gas
                if Cfit[j][i] < fitCB[j]:
                    fitCB[j] = Cfit[j][i]
                    Cxb[j, :] = Cx[j][i, :]
                # Update global best
                if Cfit[j][i] < fitG:
                    fitG = Cfit[j][i]
                    Xgb = Cx[j][i, :]
        # Pre
        S = np.zeros((num_gas, Nn))

        curve = np.zeros(self.max_Iter)
        curve[0] = fitG
        t = 1
        # Iterations
        while t < self.max_Iter:
            # Compute temperature (8)
            T = math.exp(-t / self.max_Iter)
            for j in range(num_gas):
                # Update henry coefficient (8)
                H[j] = H[j] * math.exp(-C[j] * ((1 / T) - (1 / Ttheta)))
                for i in range(Cx[j].shape[0]):
                    # Update solubility (9)
                    S[j, i] = K * H[j] * P[j, i]
                    # Compute gamma (10)
                    gamma = self.beta * math.exp(-((fitG + eps) / (Cfit[j][i] + eps)))
                    # Flag change between - & +
                    if random.random() > 0.5:
                        F = -1
                    else:
                        F = 1
                    for d in range(dim):
                        # Random constant
                        r = random.random()
                        # Position update (10)
                        Cx[j][i, d] = Cx[j][i, d] + F * r * gamma * \
                                      (Cxb[j, d] - Cx[j][i, d]) + F * r * self.alpha * \
                                      (S[j, i] * Xgb[d] - Cx[j][i, d])
                    # Boundary
                    XB = Cx[j][i, :]
                    XB[XB > ub] = ub
                    XB[XB < lb] = lb
                    Cx[j][i, :] = XB
            # Fitness
            for j in range(num_gas):
                for i in range(Cx[j].shape[0]):
                    # Fitness
                    Cfit[j][i] = self.loss_func(x_train[:, Cx[j][i, :] > self.thres], x_test[:, Cx[j][i, :] > self.thres], y_train, y_test)

            # Select the worst solution (11)
            Nw = round(N * (random.random() * (c2 - c1) + c1))
            # Convert cell to array
            XX = np.concatenate(Cx)
            FF = np.concatenate(Cfit)
            idx = np.argsort(FF)
            # Update position of worst solution (12)
            for i in range(Nw):
                for d in range(dim):
                    XX[idx[i], d] = lb + random.random() * (ub - lb)
                # Fitness
                FF[idx[i]] = self.loss_func(x_train[:, XX[idx][i, :] > self.thres], x_test[:, XX[idx][i, :] > self.thres], y_train, y_test)

            # Divide the population into Nc type of gas cluster back
            for j in range(num_gas):
                if j != num_gas:
                    Cx[j] = XX[(j - 1) * Nn: j * Nn, :]
                    Cfit[j] = FF[(j - 1) * Nn: j * Nn]
                else:
                    Cx[j] = XX[(num_gas - 1) * Nn:, :]
                    Cfit[j] = FF[(num_gas - 1) * Nn:]

            # Update best solution
            for j in range(num_gas):
                for i in range(Cx[j].shape[0]):
                    # Update best gas
                    if Cfit[j][i] < fitCB[j]:
                        fitCB[j] = Cfit[j][i]
                        Cxb[j] = Cx[j][i]
                    # Update global best
                    if Cfit[j][i] < fitG:
                        fitG = Cfit[j][i]
                        Xgb = Cx[j][i]
            curve[t - 1] = fitG
            print('\nIteration %d Best (HGSO)= %f' % (t, curve[t - 1]))
            t = t + 1

        # Select features
        Pos = np.arange(dim)
        Sf = Pos[Xgb > self.thres]
        # Store results
        HGSO = {}
        HGSO['sf'] = Sf
        HGSO['c'] = curve

        return HGSO