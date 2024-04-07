import numpy as np

class jAntLionOptimizer:
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


        # Number of dimensions
        dim = x_train.shape[1]

        # Initial: Ant & antlion
        Xal = np.random.uniform(lb, ub, size=(self.N, dim))
        Xa = np.random.uniform(lb, ub, size=(self.N, dim))

        # Fitness of antlion
        fitAL = np.zeros(self.N)
        fitE = np.inf
        for i in range(self.N):
            fitAL[i] = self.loss_func(x_train[:, Xal[i, :] > self.thres], x_test[:, Xal[i, :] > self.thres],
                                    y_train, y_test)
            # Elite update
            if fitAL[i] < fitE:
                Xe = Xal[i, :]
                fitE = fitAL[i]

        # Pre
        fitA = np.ones(self.N)

        curve = np.zeros(self.max_Iter)
        curve[0] = fitE
        t = 2

        # Iteration
        while t <= self.max_Iter:
            # Set weight according to iteration
            I = 1
            if t > 0.1 * self.max_Iter:
                w = 2
                I = (10 ** w) * (t / self.max_Iter)
            elif t > 0.5 * self.max_Iter:
                w = 3
                I = (10 ** w) * (t / self.max_Iter)
            elif t > 0.75 * self.max_Iter:
                w = 4
                I = (10 ** w) * (t / self.max_Iter)
            elif t > 0.9 * self.max_Iter:
                w = 5
                I = (10 ** w) * (t / self.max_Iter)
            elif t > 0.95 * self.max_Iter:
                w = 6
                I = (10 ** w) * (t / self.max_Iter)

            # Radius of ant's random walks hyper-sphere (2.10-2.11)
            c = lb / I
            d = ub / I

            # Convert probability
            Ifit = 1 / (1 + fitAL)
            prob = Ifit / np.sum(Ifit)

            for i in range(self.N):
                # Select one antlion using roulette wheel
                rs = self._roulette_wheel_selection(prob)
                # Apply random walk of ant around antlion
                RA = self._random_walk_ALO(Xal[rs, :], c, d, self.max_Iter, dim)
                # Apply random walk of ant around elite
                RE = self._random_walk_ALO(Xe, c, d, self.max_Iter, dim)
                # Elitism process (2.13)
                Xa[i, :] = (RA[t, :] + RE[t, :]) / 2
                # Boundary
                XB = Xa[i, :]
                XB[XB > ub] = ub
                XB[XB < lb] = lb
                Xa[i, :] = XB

            # Fitness
            for i in range(self.N):
                # Fitness of ant
                fitA[i] = self.loss_func(x_train[:, Xa[i, :] > self.thres], x_test[:, Xa[i, :] > self.thres],
                                          y_train, y_test)
                # Elite update
                if fitA[i] < fitE:
                    Xe = Xa[i, :]
                    fitE = fitA[i]

            # Update antlion position
            XX = np.vstack((Xal, Xa))
            FF = np.concatenate((fitAL, fitA))
            idx = np.argsort(FF)
            Xal = XX[idx[:self.N], :]
            fitAL = FF[:self.N]

            # Save
            curve[t-1] = fitE
            print('\nIteration %d Best (ALO) = %f' % (t, curve[t-1]))
            t = t + 1

        # Select features
        Pos = np.arange(dim)
        Sf = Pos[Xe > self.thres]
        sFeat = x_train[:, Sf]

        # Store results
        ALO = {}
        ALO['sf'] = Sf.tolist()
        ALO['c'] = curve.tolist()

        return ALO

    @staticmethod
    def _roulette_wheel_selection(prob):
        # Cummulative summation
        C = np.cumsum(prob)
        # Random one value, most probability value [0~1]
        P = np.random.rand()
        # Route wheel
        for i in range(len(C)):
            if C[i] > P:
                Index = i
                break
        return Index

    @staticmethod
    def _random_walk_ALO(Xal, c, d, max_Iter, dim):
        # Pre
        RW = np.zeros((max_Iter + 1, dim))
        R = np.zeros(max_Iter)
        # Random walk with C on antlion (2.8)
        if np.random.rand() > 0.5:
            c = Xal + c
        else:
            c = Xal - c
        # Random walk with D on antlion (2.9)
        if np.random.rand() > 0.5:
            d = Xal + d
        else:
            d = Xal - d

        for j in range(dim):
            # Random distribution (2.2)
            for t in range(max_Iter):
                if np.random.rand() > 0.5:
                    R[t] = 1
                else:
                    R[t] = 0
            # Actual random walk (2.1)
            X = np.concatenate(([0], np.cumsum((2 * R) - 1)))
            # [a,b]-->[c,d]
            a = np.min(X)
            b = np.max(X)
            # Normalized (2.7)
            Xnorm = (((X - a) * (d[j] - c[j])) / (b - a)) + c[j]
            # Store result
            RW[:, j] = Xnorm

        return RW
