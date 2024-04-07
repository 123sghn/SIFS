import numpy as np
from losses.jFitnessFunction import jFitnessFunction

class jWildHorseOptimizer:
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

        dim = x_train.shape[1]
        lb = 0
        ub = 1
        P = 0.5
        FADs = 0.2
        N = self.N

        curve = np.zeros(self.max_Iter)
        X = lb + (ub - lb) * np.random.rand(N, dim)
        fit = np.zeros(N)
        for i in range(N):
            fit[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)
        fit = np.array(fit)
        ind = np.argsort(fit)
        X = X[ind,:]
        best_fit = fit[0]
        best_X = X[0,:]

        for iter in range(self.max_Iter):
            for i in range(1, N):
                r = np.random.rand()
                if r < 0.5:
                    X[i,:] = X[i,:] * np.random.randint(0, 2, size=X.shape[1])

            X[X > ub] = ub
            X[X < lb] = lb

            for i in range(1, N):
                fit[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)

            fit = np.array(fit)
            ind = np.argsort(fit)
            X = X[ind,:]

            if fit[0] < best_fit:
                best_fit = fit[0]
                best_X = X[0,:]

            for i in range(1, N):
                r = np.random.rand()
                if r < 0.5:
                    dist = np.sum(np.logical_xor(X,X[i,:]), axis=1)
                    ind = np.argsort(dist)
                    partner = X[ind[1],:]
                    mask = np.random.randint(0, 2, size=partner.shape)
                    offspring = np.logical_and(X[i,:], np.logical_not(mask)) | np.logical_and(partner, mask)
                    offspring[offspring > ub] = ub
                    offspring[offspring < lb] = lb
                    offspring_fit = self.loss_func(x_train[:, offspring > self.thres], x_test[:, offspring > self.thres],
                                        y_train, y_test)
                    if offspring_fit < fit[i]:
                        X[i,:] = offspring
                        fit[i] = offspring_fit
                        if fit[i] < best_fit:
                            best_fit = fit[i]
                            best_X = X[i,:]

            curve[iter] = best_fit
            print('Iteration {} Best (WHO)= {}'.format(iter, curve[iter]))

        Sf = np.where(best_X > self.thres)[0]
        WHO = {'sf':Sf, 'c':curve}

        print('Final best fitness is', best_fit)
        print('Final best solution is', best_X)
        return WHO
