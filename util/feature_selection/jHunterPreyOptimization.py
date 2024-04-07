import numpy as np
from losses.jFitnessFunction import jFitnessFunction


class jHunterPreyOptimization:
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
        B=0.7,
    ):
        self.N = N
        self.max_Iter = max_Iter
        self.loss_func = loss_func
        self.alpha = alpha
        self.beta = beta
        self.thres = thres
        self.tau = tau
        self.rho = rho
        self.eta = eta
        self.B = B

    def optimize(self, x_train, x_test, y_train, y_test):

        dim = x_train.shape[1]
        lb = 0
        ub = 1
        N = self.N
        B = self.B

        # HPO Parameters
        curve = np.zeros(self.max_Iter)
        # Initialization
        X = np.random.rand(N, dim) * (ub - lb) + lb
        # Evaluate
        fitness = np.zeros(N)
        for i in range(N):
            fitness[i] = self.loss_func(
                x_train[:, X[i, :] > self.thres],
                x_test[:, X[i, :] > self.thres],
                y_train,
                y_test,
            )

        indx = np.argmin(fitness)
        #
        Xgb = X[indx, :].copy()  # Xgb HPO
        fitG = fitness[indx]
        curve[0] = fitG

        # HPO Main Loop
        for t in range(1, self.max_Iter):
            c = 1 - t * ((0.98) / self.max_Iter)  # Update C Parameter
            kbest = int(N * c)  # Update kbest
            for i in range(N):
                r1 = np.random.rand(dim) < c
                r2 = np.random.rand()
                r3 = np.random.rand(dim)
                idx = r1 == 0
                z = r2 * idx + r3 * ~idx
                if np.random.rand() < B:
                    xi = np.mean(X, axis=0)
                    dist = np.linalg.norm(xi - X, axis=1)
                    idxsortdist = np.argsort(dist)
                    SI = X[idxsortdist[kbest], :]
                    X[i, :] = X[i, :] + 0.5 * (
                        (2 * (c) * z * SI - X[i, :]) + (2 * (1 - c) * z * xi - X[i, :])
                    )
                else:
                    for j in range(dim):
                        rr = -1 + 2 * z[j]
                        X[i, j] = (
                            2 * z[j] * np.cos(2 * np.pi * rr) * (Xgb[j] - X[i, j])
                            + Xgb[j]
                        )
                X[i, :] = np.clip(X[i, :], lb, ub)
                # Evaluation
                fitness[i] = self.loss_func(
                    x_train[:, X[i, :] > self.thres],
                    x_test[:, X[i, :] > self.thres],
                    y_train,
                    y_test,
                )

                # Update Xgb
                if fitness[i] < fitG:
                    Xgb = X[i, :].copy()
                    fitG = fitness[i]
            curve[t] = fitG
            print("\nIteration %d Best (HPO)= %f" % (t, curve[t]))

        Pos = np.arange(1, dim + 1)
        Sf = Pos[(Xgb > self.thres) == 1] - 1

        HPO = {}
        HPO["sf"] = Sf  # best_X
        HPO["c"] = curve
        return HPO
