import numpy as np

from losses.jFitnessFunction import jFitnessFunction

class jTreeGrowthAlgorithm:
    def __init__(self, N, max_Iter, loss_func, alpha=0.9, beta=0.1, thres=0.5, tau=1, rho=0.2, eta=1, num_tree1=3, num_tree2=3, num_tree4=3, theta=0.8, lambda_=0.5):
        self.N = N
        self.max_Iter = max_Iter
        self.loss_func = loss_func
        self.alpha = alpha
        self.beta = beta
        self.thres = thres
        self.tau = tau
        self.rho = rho
        self.eta = eta
        self.num_tree1=num_tree1
        self.num_tree2=num_tree2
        self.num_tree4=num_tree4
        self.theta = theta
        self.lambda_=lambda_

    def optimize(self, x_train, x_test, y_train, y_test):

        # Parameters
        lb = 0
        ub = 1
        num_tree1 = self.num_tree1  # size of first group
        num_tree2 = self.num_tree2  # size of second group
        num_tree4 = self.num_tree4  # size of fourth group
        theta = self.theta  # tree reduction rate of power
        lambda_ = self.lambda_  # control nearest tree
        N = self.N

        # Limit number of N4 to N1
        if num_tree4 > num_tree1 + num_tree2:
            num_tree4 = num_tree1 + num_tree2


        # Number of dimensions
        dim = x_train.shape[1]

        # Initial
        X = np.zeros((N, dim))
        for i in range(N):
            X[i] = lb + (ub - lb) * np.random.rand(dim)

        # Fitness
        fit = np.zeros(N)
        fitG = np.inf
        for i in range(N):
            fit[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)
            # Best
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i]

        # Sort tree from best to worst
        idx = np.argsort(fit)
        fit = fit[idx]
        X = X[idx]

        # Initial
        dist = np.zeros(num_tree1 + num_tree2)
        X1 = np.zeros((num_tree1, dim))
        Xnew = np.zeros((num_tree4, dim))
        Fnew = np.zeros(num_tree4)

        curve = np.zeros(self.max_Iter)
        curve[0] = fitG
        t = 1

        # Iterations
        while t < self.max_Iter:
            # {1} Best trees group
            for i in range(num_tree1):
                r1 = np.random.rand()
                for d in range(dim):
                    # Local search (1)
                    X1[i, d] = (X[i, d] / theta) + r1 * X[i, d]
                # Boundary
                XB = X1[i, :]
                XB[XB > ub] = ub
                XB[XB < lb] = lb
                X1[i, :] = XB
                # Fitness
                fitT = self.loss_func(x_train[:, X1[i, :] > self.thres], x_test[:, X1[i, :] > self.thres],
                                        y_train, y_test)
                # Greedy selection
                if fitT <= fit[i]:
                    X[i, :] = X1[i, :]
                    fit[i] = fitT

            # {2} Competitive for light tree group
            X_ori = X
            for i in range(num_tree1 + num_tree2, num_tree1 + num_tree2 + num_tree2):
                # Neighbor tree
                for j in range(num_tree1 + num_tree2):
                    if j != i:
                        # Compute Euclidean distance (2)

                        dist[j] = np.sqrt(np.sum((X_ori[j, :] - X_ori[i, :]) ** 2))
                    else:
                        # Solve same tree problem
                        dist[j] = np.inf
                # Find 2 trees with shorter distance
                idx = np.argsort(dist)
                T1 = X_ori[idx[0], :]
                T2 = X_ori[idx[1], :]
                # Alpha in [0,1]
                alpha = np.random.rand()
                for d in range(dim):
                    # Compute linear combination between 2 shorter tree (3)
                    y = lambda_ * T1[d] + (1 - lambda_) * T2[d]
                    # Move tree i between 2 adjacent trees (4)
                    X[i, d] = X[i, d] + alpha * y
                # Boundary
                XB = X[i, :]
                XB[XB > ub] = ub
                XB[XB < lb] = lb
                X[i, :] = XB
                # Fitness
                fit[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)

            # {3} Remove and replace group
            for i in range(num_tree1 + num_tree2 + num_tree2, N):
                X[i] = lb + (ub - lb) * np.random.rand(dim)
                # Fitness
                fit[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)

            # {4} Reproduction group
            for i in range(num_tree4):
                # Random a best tree
                r = np.random.randint(1, num_tree1)
                Xbest = X[r]
                # Mask operator
                mask = np.random.randint(0, 2, dim)
                # Mask opration between new & best trees
                for d in range(dim):
                    # Generate new solution
                    Xn = lb + (ub - lb) * np.random.rand()
                    if mask[d] == 1:
                        Xnew[i, d] = Xbest[d]
                    elif mask[d] == 0:
                        # Generate new tree
                        Xnew[i, d] = Xn
                # Fitness
                Fnew[i] = self.loss_func(x_train[:, Xnew[i, :] > self.thres], x_test[:, Xnew[i, :] > self.thres],
                                        y_train, y_test)

            # Sort population get best nPop trees
            XX = np.concatenate((X, Xnew), axis=0)
            FF = np.concatenate((fit, Fnew))
            idx = np.argsort(FF)
            X = XX[idx[:N]]
            fit = FF[:N]

            # Global best
            if fit[0] < fitG:
                fitG = fit[0]
                Xgb = X[0]

            curve[t] = fitG
            print('\nIteration', t, 'Best (TGA) =', curve[t])
            t += 1

        # Select features
        Pos = np.arange(dim)
        Sf = Pos[Xgb > self.thres]

        # Store results
        TGA = {}
        TGA['sf'] = Sf
        TGA['c'] = curve

        return TGA