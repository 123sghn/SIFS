import numpy as np

from losses.jFitnessFunction import jFitnessFunction

class jHumanLearningOptimization:
    def __init__(self, N, max_Iter, loss_func, alpha=0.9, beta=0.1, thres=0.5, tau=1, rho=0.2, eta=1, pi=0.85, pr=0.1):
        self.N = N
        self.max_Iter = max_Iter
        self.loss_func = loss_func
        self.alpha = alpha
        self.beta = beta
        self.thres = thres
        self.tau = tau
        self.rho = rho
        self.eta = eta
        self.pi = pi
        self.pr = pr

    def optimize(self, x_train, x_test, y_train, y_test):

        # Parameters
        pi = self.pi   # probability of individual learning
        pr = self.pr    # probability of exploration learning
        N = self.N


        # Number of dimensions
        dim = x_train.shape[1]

        # Initial
        X = self._Initial_Population(N, dim)

        # Fitness
        fit = np.zeros(N)
        fitSKD = np.inf
        for i in range(N):
            fit[i] = self.loss_func(x_train[:,X[i, :] > self.thres], x_test[:,X[i, :] > self.thres], y_train, y_test)

            # Update SKD/gbest
            if fit[i] < fitSKD:
                fitSKD = fit[i]
                SKD = X[i, :]

        # Get IKD/pbest
        fitIKD = fit.copy()
        IKD = X.copy()

        # Pre
        curve = np.zeros(self.max_Iter)
        curve[0] = fitSKD
        t = 2

        # Generations
        while t <= self.max_Iter:
            for i in range(N):
                # Update solution (8)
                for d in range(dim):
                    # Radom probability in [0,1]
                    r = np.random.rand()
                    if r >= 0 and r < pr:
                        # Random exploration learning operator (7)
                        if np.random.rand() < 0.5:
                            X[i, d] = 0
                        else:
                            X[i, d] = 1
                    elif r >= pr and r < pi:
                        X[i, d] = IKD[i, d]
                    else:
                        X[i, d] = SKD[d]

            # Fitness
            for i in range(N):
                # Fitness
                fit[i] = self.loss_func(x_train[:,X[i, :] > self.thres], x_test[:,X[i, :] > self.thres], y_train, y_test)

                # Update IKD/pbest
                if fit[i] < fitIKD[i]:
                    fitIKD[i] = fit[i]
                    IKD[i, :] = X[i, :]
                # Update SKD/gbest
                if fitIKD[i] < fitSKD:
                    fitSKD = fitIKD[i]
                    SKD = IKD[i, :]

            curve[t-1] = fitSKD
            print('\nGeneration %d Best (HLO)= %f' % (t, curve[t-1]))
            t = t + 1

        # Select features based on selected index
        Pos = np.arange(dim)
        Sf = Pos[SKD == 1]

        # Store results
        HLO = {}
        HLO['sf'] = Sf.tolist()
        HLO['c'] = curve.tolist()

        return HLO

    @staticmethod
    # Binary initialization strategy
    def _Initial_Population(N, dim):
        X = np.zeros((N, dim))
        for i in range(N):
            for d in range(dim):
                if np.random.rand() > 0.5:
                    X[i, d] = 1
        return X
