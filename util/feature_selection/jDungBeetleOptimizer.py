import numpy as np
from losses.jFitnessFunction import jFitnessFunction


class jDungBeetleOptimizer:
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

        import numpy as np

        N = self.N
        dim = x_train.shape[1]  # dimensio
        curve = np.zeros(self.max_Iter)
        fun = jFitnessFunction  # Name of the Function
        P_percent = 0.2  # The population size of producers accounts for "P_percent" percent of the total population size
        pNum = round(N * P_percent)  # The population size of the producers
        lb = (0) * np.ones(dim)  # Upper Bound
        ub = (1) * np.ones(dim)  # Lower Bound
        X = np.random.rand(N, dim) * (ub - lb) + lb
        fit = np.zeros(N)
        # Initialization
        for i in range(N):
            fit[i] = self.loss_func(
                x_train[:, X[i, :] > self.thres],
                x_test[:, X[i, :] > self.thres],
                y_train,
                y_test,
            )

        pFit = fit
        pX = X
        XX = pX
        fitG, bestI = np.min(fit), np.argmin(
            fit
        )  # fitG denotes the global optimum fitness value
        Xgb = X[
            bestI, :
        ]  # Xgb denotes the global optimum position corresponding to fitG

        # Start updating the solutions.
        import numpy as np

        for t in range(self.max_Iter):
            fmax, B = np.max(fit), np.argmax(fit)
            worse = X[B, :]
            r2 = np.random.rand()

            for i in range(pNum):
                if r2 < 0.9:
                    r1 = np.random.rand()
                    a = np.random.rand()
                    if a > 0.1:
                        a = 1
                    else:
                        a = -1
                    X[i, :] = (
                        pX[i, :] + 0.3 * np.abs(pX[i, :] - worse) + a * 0.1 * (XX[i, :])
                    )  # Equation (1)
                else:
                    aaa = np.random.permutation(180)[0]
                    if aaa == 0 or aaa == 90 or aaa == 180:
                        X[i, :] = pX[i, :]
                    theta = aaa * np.pi / 180
                    X[i, :] = pX[i, :] + np.tan(theta) * np.abs(
                        pX[i, :] - XX[i, :]
                    )  # Equation (2)

                X[i, :] = self._bounds(X[i, :], lb, ub)
                fit[i] = self.loss_func(
                    x_train[:, X[i, :] > self.thres],
                    x_test[:, X[i, :] > self.thres],
                    y_train,
                    y_test,
                )

            fMMin, bestII = np.min(fit), np.argmin(fit)
            bestXX = X[bestII, :]
            R = 1 - t / self.max_Iter

            Xnew1 = bestXX * (1 - R)
            Xnew2 = bestXX * (1 + R)  # Equation (3)
            Xnew1 = self._bounds(Xnew1, lb, ub)
            Xnew2 = self._bounds(Xnew2, lb, ub)

            Xnew11 = Xgb * (1 - R)
            Xnew22 = Xgb * (1 + R)  # Equation (5)
            Xnew11 = self._bounds(Xnew11, lb, ub)
            Xnew22 = self._bounds(Xnew22, lb, ub)

            for i in range(pNum + 1, 12):  # Equation (4)
                X[i, :] = bestXX + (
                    np.random.rand(dim) * (pX[i, :] - Xnew1)
                    + np.random.rand(dim) * (pX[i, :] - Xnew2)
                )
                X[i, :] = self._bounds(X[i, :], Xnew1, Xnew2)
                fit[i] = self.loss_func(
                    x_train[:, X[i, :] > self.thres],
                    x_test[:, X[i, :] > self.thres],
                    y_train,
                    y_test,
                )

            for i in range(13, 19):  # Equation (6)
                X[i, :] = pX[i, :] + (
                    np.random.randn(1) * (pX[i, :] - Xnew11)
                    + (np.random.rand(dim) * (pX[i, :] - Xnew22))
                )
                X[i, :] = self._bounds(X[i, :], lb, ub)
                fit[i] = self.loss_func(
                    x_train[:, X[i, :] > self.thres],
                    x_test[:, X[i, :] > self.thres],
                    y_train,
                    y_test,
                )

            for j in range(20, N):  # Equation (7)
                X[j, :] = (
                    Xgb
                    + np.random.randn(dim)
                    * ((np.abs(pX[j, :] - bestXX)) + (np.abs(pX[j, :] - Xgb)))
                    / 2
                )
                X[j, :] = self._bounds(X[j, :], lb, ub)
                fit[j] = self.loss_func(
                    x_train[:, X[i, :] > self.thres],
                    x_test[:, X[i, :] > self.thres],
                    y_train,
                    y_test,
                )

            XX = pX
            for i in range(N):
                if fit[i] < pFit[i]:
                    pFit[i] = fit[i]
                    pX[i, :] = X[i, :]
                if pFit[i] < fitG:
                    fitG = pFit[i]
                    Xgb = pX[i, :]

            curve[t] = fitG
            print("\nIteration {} Best (DBO)= {}".format(t, curve[t]))
        Pos = np.arange(1, dim + 1)
        Sf = Pos[(Xgb > self.thres) == 1] - 1

        DBO = {}
        DBO["sf"] = Sf  # best_X
        DBO["c"] = curve
        return DBO

    @staticmethod
    def _bounds(SS, LLb, UUb):
        # Apply the lower bound vector
        temp = SS.copy()
        I = temp < LLb
        temp[I] = LLb[I]
        # Apply the upper bound vector
        J = temp > UUb
        temp[J] = UUb[J]

        # Update this new move
        S = temp.copy()
        return S
