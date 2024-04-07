from typing import List, Dict
import numpy as np
from losses.jFitnessFunction import jFitnessFunction

class jAntColonySystem:
    def __init__(self, N, max_Iter, loss_func, alpha=0.9, beta=0.1, thres=0.5, tau=1, rho=0.2, eta=1, phi=0.5):
        self.N = N
        self.max_Iter = max_Iter
        self.loss_func = loss_func
        self.alpha = alpha
        self.beta = beta
        self.thres = thres
        self.tau = tau
        self.rho = rho
        self.eta = eta
        self.phi = phi

    def optimize(self, x_train, x_test, y_train, y_test):
        feat = np.concatenate((x_train, x_test), axis=0)
        label = np.concatenate((y_train, y_test), axis=0)

        # Parameters
        tau = self.tau  # pheromone value
        eta = self.eta  # heuristic desirability
        alpha = self.alpha  # control pheromone
        beta = self.beta  # control heuristic
        rho = self.rho  # pheromone trail decay coefficient
        phi = self.phi  # pheromena coefficient
        # Number of dimensions
        dim = feat.shape[1]
        # Initial Tau & Eta
        tau = tau * np.ones((dim, dim))
        eta = eta * np.ones((dim, dim))
        # Pre
        fitG = np.inf
        fit = np.zeros(self.N)
        tau0 = tau

        curve = np.zeros(self.max_Iter)
        t = 1
        # Iterations
        while t <= self.max_Iter:
            # Reset ant
            X = np.zeros((self.N, dim))
            for i in range(self.N):
                # Set number of features
                num_feat = np.random.randint(1, dim + 1)
                # Ant start with random position
                X[i, 0] = np.random.randint(1, dim + 1)
                k = []
                if num_feat > 1:
                    for d in range(1, num_feat):
                        # Start with previous tour
                        k = np.append(k, X[i, d - 1])
                        # Edge / Probability Selection (4)
                        P = (tau[int(k[-1]) - 1, :] ** alpha) * (eta[int(k[-1]) - 1, :] ** beta)
                        # Set selected position = 0 probability (4)
                        P[k.astype(int) - 1] = 0
                        # Convert probability (4)
                        prob = P / np.sum(P)
                        # Roulette Wheel selection
                        route = self._roulette_wheel_selection(prob)
                        # Store selected position to be next tour
                        X[i, d] = route
            # Binary
            X_bin = np.zeros((self.N, dim))
            for i in range(self.N):
                # Binary form
                ind = X[i, :].astype(int)
                ind = ind[ind != 0]
                X_bin[i, ind - 1] = 1
            # Binary version
            for i in range(self.N):
                # Fitness
                fit[i] = self.loss_func(x_train[:, X_bin[i, :] > self.thres], x_test[:, X_bin[i, :] > self.thres],
                                        y_train, y_test)
                # Global update
                if fit[i] < fitG:
                    Xgb = X[i, :]
                    fitG = fit[i]
            # Tau update
            tour = Xgb
            tour = tour[tour != 0]
            tour = np.append(tour, tour[0])
            for d in range(len(tour) - 1):
                # Feature selected
                x = int(tour[d]) - 1
                y = int(tour[d + 1]) - 1
                # Delta tau
                Dtau = 1 / fitG
                # Update tau (10)
                tau[x, y] = (1 - phi) * tau[x, y] + phi * Dtau
            # Evaporate pheromone (9)
            tau = (1 - rho) * tau + rho * tau0
            # Save
            curve[t - 1] = fitG
            print('\nIteration %d Best (ACS)= %f' % (t, curve[t - 1]))
            t = t + 1

        # Select features based on selected index
        Sf = np.unique(Xgb).astype(int)
        Sf = Sf[Sf != 0] - 1

        # Store results
        ACS = {}
        ACS['sf'] = Sf
        ACS['nf'] = len(Sf)
        ACS['c'] = curve


        return ACS

    @staticmethod
    def _roulette_wheel_selection(prob: np.ndarray) -> int:
        # Cummulative summation
        C = np.cumsum(prob)
        # Random one value, most probability value [0~1]
        P = np.random.rand()
        # Route wheel
        for i in range(len(C)):
            if C[i] > P:
                Index = i + 1
                break
        return Index
