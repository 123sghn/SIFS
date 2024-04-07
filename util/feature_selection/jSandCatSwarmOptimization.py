import numpy as np
from losses.jFitnessFunction import jFitnessFunction


class jSandCatSwarmOptimization:
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
        P = 0.5  # constant
        FADs = 0.2  # fish aggregating devices effect
        N = self.N

        Xgb = np.zeros(dim)
        fitG = np.inf

        X = np.random.rand(N, dim) * (ub - lb) + lb
        curve = np.zeros(self.max_Iter)
        t = 0
        p = np.arange(1, 361)
        while t < self.max_Iter:
            for i in range(X.shape[0]):
                Flag4ub = X[i, :] > ub
                Flag4lb = X[i, :] < lb
                # 更新X
                X[i, :] = (
                    (X[i, :] * (~(Flag4ub + Flag4lb))) + ub * Flag4ub + lb * Flag4lb
                )

                fitness = self.loss_func(
                    x_train[:, X[i, :] > self.thres],
                    x_test[:, X[i, :] > self.thres],
                    y_train,
                    y_test,
                )
                if fitness < fitG:
                    fitG = fitness
                    Xgb = X[i, :].copy()

            S = 2  # S is maximum Sensitivity range
            rg = S - ((S) * t / (self.max_Iter))  # guides R
            for i in range(X.shape[0]):
                r = np.random.rand() * rg
                R = ((2 * rg) * np.random.rand()) - rg  # controls to transtion phases
                for j in range(X.shape[1]):
                    teta = self._roulette_wheel_Selection(p)
                    if (-1 <= R) and (R <= 1):  # R value is between -1 and 1
                        Rand_position = np.abs(np.random.rand() * Xgb[j] - X[i, j])
                        X[i, j] = Xgb[j] - r * Rand_position * np.cos(teta)
                    else:
                        cp = np.floor(N * np.random.rand()).astype(int)
                        CandidatePosition = X[cp, :]
                        X[i, j] = r * (
                            CandidatePosition[j] - np.random.rand() * X[i, j]
                        )

            t += 1
            curve[t - 1] = fitG
            print(f"\nIteration {t} Best (SCSO) = {curve[t-1]}")

        Pos = np.arange(1, dim + 1)
        Sf = Pos[Xgb > self.thres]  # selected features

        SCSO = {"sf": Sf, "c": curve}

        return SCSO

    @staticmethod
    def _roulette_wheel_Selection(P):
        r = np.random.rand()
        s = np.sum(P)
        P = P / s
        C = np.cumsum(P)
        j = np.where(r <= C)[0][0]
        return j
