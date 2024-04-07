from losses.jFitnessFunction import jFitnessFunction


class jChameleonSwarmAlgorithm:
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

    def optimize(self, x_train, x_test, y_train, y_test):

        import numpy as np

        dim = x_train.shape[1]
        lb = 0
        ub = 1
        N = self.N

        ub = np.ones(dim) * ub
        lb = np.ones(dim) * lb

        # if len(ub.shape[0]) == 1:
        #     ub = np.tile(ub, (dim,))
        #     lb = np.tile(lb, (dim,))

        # Convergence curve
        curve = np.zeros(self.max_Iter)

        # Initial population
        X = np.random.rand(N, dim) * (ub - lb) + lb
        # Evaluate the fitness of the initial population
        fit = np.zeros((N,))
        for i in range(N):
            fit[i] = self.loss_func(
                x_train[:, X[i, :] > self.thres],
                x_test[:, X[i, :] > self.thres],
                y_train,
                y_test,
            )
        # Initalize the parameters of CSA
        fitness = fit
        fitG, index = np.min(fit), np.argmin(fit)
        chameleonBestPosition = X.copy()
        Xgb = X[index].copy()
        v = 0.1 * chameleonBestPosition
        v0 = 0.0 * v

        # Start CSA
        # Main parameters of CSA
        rho = 1.0
        gamma = 2.0
        alpha = 4.0
        beta = 3.0

        import numpy as np

        # Start CSA
        for t in range(1, self.max_Iter + 1):
            a = 2590 * (1 - np.exp(-np.log(t)))
            omega = (1 - (t / self.max_Iter)) ** (rho * np.sqrt(t / self.max_Iter))
            p1 = 2 * np.exp(-2 * (t / self.max_Iter) ** 2)
            p2 = 2 / (1 + np.exp((-t + self.max_Iter / 2) / 100))

            mu = gamma * np.exp(-((alpha * t / self.max_Iter) ** beta))

            ch = np.ceil(N * np.random.rand(N))

            ## Update the position of CSA (Exploration)
            for i in range(N):
                if np.random.rand() >= 0.1:
                    X[i, :] = (
                        X[i, :]
                        + p1
                        * (chameleonBestPosition[int(ch[i]) - 1, :] - X[i, :])
                        * np.random.rand()
                        + p2 * (Xgb - X[i, :]) * np.random.rand()
                    )
                else:
                    for j in range(dim):
                        X[i, j] = Xgb[j] + mu * (
                            (ub[j] - lb[j]) * np.random.rand() + lb[j]
                        ) * np.sign(np.random.rand() - 0.50)

            # Rotation of the chameleons - Update the position of CSA (Exploitation)
            # [X] = rotation(X, N, dim);
            # Code for rotation function

            # Chameleon velocity updates and find a food source
            for i in range(N):
                v[i, :] = (
                    omega * v[i, :]
                    + p1 * (chameleonBestPosition[i, :] - X[i, :]) * np.random.rand()
                    + p2 * (Xgb - X[i, :]) * np.random.rand()
                )
                X[i, :] = X[i, :] + (v[i, :] ** 2 - v0[i, :] ** 2) / (2 * a)

            v0 = v

            # Handling boundary violations
            for i in range(N):
                for j in range(dim):
                    if X[i, j] < lb[j]:
                        X[i, j] = lb[j]
                    elif X[i, j] > ub[j]:
                        X[i, j] = ub[j]

            # Relocation of chameleon positions (Randomization)
            for i in range(N):
                ub_ = np.sign(X[i, :] - ub) > 0
                lb_ = np.sign(X[i, :] - lb) < 0
                X[i, :] = (X[i, :] * (~np.logical_xor(lb_, ub_))) + ub * ub_ + lb * lb_

                fit = self.loss_func(
                    x_train[:, X[i, :] > self.thres],
                    x_test[:, X[i, :] > self.thres],
                    y_train,
                    y_test,
                )

                if fit < fitness[i]:
                    chameleonBestPosition[i, :] = X[i, :]  # Update the best positions
                    fitness[i] = fit  # Update the fitness

            # Evaluate the new positions
            fmin, index = np.min(fitness), np.argmin(fitness)

            # Updating Xgb and best fitness
            if fmin < fitG:
                Xgb = chameleonBestPosition[
                    index, :
                ]  # Update the global best positions
                fitG = fmin

            # Visualize the results
            curve[t - 1] = fitG  # Best found value until iteration t
            print(f"\nIteration {t} Best (CSA) = {fitG}")

        Pos = np.arange(1, dim + 1)
        Sf = Pos[(Xgb > self.thres) == 1] - 1

        CSA = {}
        CSA["sf"] = Sf  # best_X
        CSA["c"] = curve
        return CSA
