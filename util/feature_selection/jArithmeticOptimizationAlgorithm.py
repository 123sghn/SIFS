import numpy as np

from losses.jFitnessFunction import jFitnessFunction


class jArithmeticOptimizationAlgorithm:
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
        Alpha=5,
        Mu=0.5,
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
        self.Alpha = Alpha
        self.Mu = Mu

    def optimize(self, x_train, x_test, y_train, y_test):
        dim = x_train.shape[1]
        lb = 0
        ub = 1
        MOP_Max = 1
        MOP_Min = 0.2
        Alpha = self.Alpha
        Mu = self.Mu

        T = 0
        curve = np.zeros(self.max_Iter)
        X = lb + (ub - lb) * np.random.rand(self.N, dim)
        Xnew = np.copy(X)
        fit = np.zeros((1, X.shape[0]))  # (fitness values)
        fitG = np.inf
        Ffun_new = np.zeros((1, Xnew.shape[0]))  # (fitness values)

        for i in range(X.shape[0]):
            fit[0, i] = self.loss_func(
                x_train[:, X[i, :] > self.thres],
                x_test[:, X[i, :] > self.thres],
                y_train,
                y_test,
            )
            if fit[0, i] < fitG:
                fitG = fit[0, i]
                Xgb = np.copy(X[i, :])

        while T < self.max_Iter:  # Main loop
            MOP = 1 - (
                (T) ** (1 / Alpha) / (self.max_Iter) ** (1 / Alpha)
            )  # Probability Ratio
            MOA = MOP_Min + T * (
                (MOP_Max - MOP_Min) / self.max_Iter
            )  # Accelerated function

            # Update the Position of solutions
            for i in range(X.shape[0]):  # if each of the ub and lb has a just value
                for j in range(X.shape[1]):
                    r1 = np.random.rand()
                    if X.shape[1] == 1:
                        if r1 < MOA:
                            r2 = np.random.rand()
                            if r2 > 0.5:
                                Xnew[i, j] = (
                                    Xgb[j]
                                    / (MOP + np.finfo(float).eps)
                                    * ((ub - lb) * Mu + lb)
                                )
                            else:
                                Xnew[i, j] = Xgb[j] * MOP * ((ub - lb) * Mu + lb)
                        else:
                            r3 = np.random.rand()
                            if r3 > 0.5:
                                Xnew[i, j] = Xgb[j] - MOP * ((ub - lb) * Mu + lb)
                            else:
                                Xnew[i, j] = Xgb[j] + MOP * ((ub - lb) * Mu + lb)
                    if (
                        X.shape[1] != 1
                    ):  # if each of the ub and lb has more than one value
                        r1 = np.random.rand()
                        if r1 < MOA:
                            r2 = np.random.rand()
                            if r2 > 0.5:
                                Xnew[i, j] = (
                                    Xgb[j]
                                    / (MOP + np.finfo(float).eps)
                                    * ((ub - lb) * Mu + lb)
                                )
                            else:
                                Xnew[i, j] = Xgb[j] * MOP * ((ub - lb) * Mu + lb)
                        else:
                            r3 = np.random.rand()
                            if r3 > 0.5:
                                Xnew[i, j] = Xgb[j] - MOP * ((ub - lb) * Mu + lb)
                            else:
                                Xnew[i, j] = Xgb[j] + MOP * ((ub - lb) * Mu + lb)

                Flag_UB = Xnew[i, :] > ub  # check if they exceed (up) the boundaries
                Flag_LB = Xnew[i, :] < lb  # check if they exceed (down) the boundaries
                Xnew[i, :] = (
                    Xnew[i, :] * (~(Flag_UB + Flag_LB)) + ub * Flag_UB + lb * Flag_LB
                )

                Ffun_new[0, i] = self.loss_func(
                    x_train[:, Xnew[i, :] > self.thres],
                    x_test[:, Xnew[i, :] > self.thres],
                    y_train,
                    y_test,
                )
                if Ffun_new[0, i] < fit[0, i]:
                    X[i, :] = Xnew[i, :]
                    fit[0, i] = Ffun_new[0, i]
                if fit[0, i] < fitG:
                    fitG = fit[0, i]
                    Xgb = np.copy(X[i, :])

            curve[T] = fitG
            print("\nIteration %d Best (AOA) = %f" % (T, curve[T - 1]))
            T += 1  # incremental iteration

        Pos = np.arange(1, dim + 1)
        Sf = Pos[(Xgb > self.thres) == 1]

        # 主循环结束
        AOA = {}
        AOA["sf"] = Sf  # Xgb
        AOA["c"] = curve

        return AOA
