import numpy as np


class jAntColonyOptimization:
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
        dim = x_train.shape[1]
        tau = self.tau * np.ones((dim, dim))
        eta = self.eta * np.ones((dim, dim))
        # Pre
        fitG = np.inf
        fit = np.zeros(self.N)

        curve = np.zeros(self.max_Iter)
        # Iterations
        for t in range(self.max_Iter):
            # Reset ant
            X = np.zeros((self.N, dim))
            for i in range(self.N):
                # Random number of features
                num_feat = np.random.randint(0, dim)
                # Ant start with random position
                X[i, 0] = np.random.randint(0, dim)
                k = np.array([])
                if num_feat > 1:
                    for d in range(1, num_feat):
                        # Start with previous tour
                        k = np.concatenate((k, [X[i, d - 1]]))
                        # Edge/Probability Selection (2)
                        P = (tau[int(k[-1]), :] ** self.alpha) * (
                            eta[int(k[-1]), :] ** self.beta
                        )
                        # Set selected position = 0 probability (2)
                        P[k.astype(int)] = 0
                        # Convert probability (2)
                        prob = P / np.sum(P)
                        # Roulette Wheel selection
                        route = jRouletteWheelSelection(prob)

                        # Store selected position to be next tour
                        X[i, d] = route
            # Binary
            X_bin = np.zeros((self.N, dim))
            for i in range(self.N):
                # Binary form
                ind = X[i, :].astype(int)
                ind = ind[ind != 0]
                X_bin[i, ind - 1] = 1
            # Fitness
            for i in range(self.N):
                # Fitness
                fit[i] = self.loss_func(
                    x_train[:, X_bin[i, :] > self.thres],
                    x_test[:, X_bin[i, :] > self.thres],
                    y_train,
                    y_test,
                )
                # Global update
                if fit[i] < fitG:
                    Xgb = X[i, :]
                    fitG = fit[i]
            # ---// [Pheromone update rule on tauK] //
            tauK = np.zeros((dim, dim))
            for i in range(self.N):
                # Update Pheromones
                tour = X[i, :]
                tour = tour[tour != 0]
                # Number of features
                len_x = len(tour)
                if len_x > 0:
                    tour = np.concatenate((tour, [tour[0]]))
                for d in range(len_x):
                    # Feature selected on graph
                    x = int(tour[d])
                    y = int(tour[d + 1])
                    # Update delta tau k on graph (3)
                    tauK[x - 1, y - 1] += 1 / (1 + fit[i])
            # ---// [Pheromone update rule on tauG] //
            tauG = np.zeros((dim, dim))
            tour = Xgb
            tour = tour[tour != 0]
            # Number of features
            len_g = len(tour)
            tour = np.concatenate((tour, [tour[0]]))
            for d in range(len_g):
                # Feature selected on graph
                x = int(tour[d])
                y = int(tour[d + 1])
                # Update delta tau G on graph
                tauG[x - 1, y - 1] = 1 / (1 + fitG)
            # ---// Evaporate pheromone // (4)
            tau = (1 - self.rho) * tau + tauK + tauG
            # Save
            curve[t] = fitG
            print("\nIteration %d Best (ACO)= %f" % (t + 1, curve[t]))

        # Select features based on selected index
        Sf = Xgb
        Sf = Sf[Sf != 0]
        # Store results
        ACO = {}
        ACO["sf"] = Sf.astype(int) - 1
        ACO["nf"] = len(Sf)
        ACO["c"] = curve

        return ACO


# // Roulette Wheel Selection //
def jRouletteWheelSelection(prob):
    sorted_indices = np.argsort(prob)[::-1]
    cum_sum = np.cumsum(prob[sorted_indices])
    P = np.random.rand()
    for i in range(len(cum_sum)):
        if cum_sum[i] > P:
            index = sorted_indices[i]
            break
    return index
