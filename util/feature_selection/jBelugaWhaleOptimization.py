import math
from losses.jFitnessFunction import jFitnessFunction
import random

import numpy as np
from scipy.special import gamma


class jBelugaWhaleOptimization:
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
        N = self.N
        fit = np.inf * np.ones((N, 1))
        newfit = fit.copy()
        curve = np.inf * np.ones(self.max_Iter)
        Counts_run = 0

        lb = lb * np.ones(dim)
        ub = ub * np.ones(dim)
        X = np.random.rand(N, dim) * (ub - lb) + lb
        for i in range(N):
            fit[i, 0] = self.loss_func(
                x_train[:, X[i, :] > self.thres],
                x_test[:, X[i, :] > self.thres],
                y_train,
                y_test,
            )
            Counts_run = Counts_run + 1
        fitG, index = np.min(fit), np.argmin(fit)
        Xgb = X[index, :]

        t = 1
        # function body goes here
        while t <= self.max_Iter:
            newpos = X.copy()
            WF = 0.1 - 0.05 * (t / self.max_Iter)  # The probability of whale fall
            kk = (1 - 0.5 * t / self.max_Iter) * np.random.rand(
                N, 1
            )  # The probability in exploration or exploitation

            for i in range(N):
                if kk[i] > 0.5:  # exploration phase
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    RJ = int(
                        np.ceil((N - 1) * np.random.rand())
                    )  # Roulette Wheel Selection
                    while RJ == i:
                        RJ = int(np.ceil((N - 1) * np.random.rand()))
                    if dim <= N / 5:
                        params = np.random.permutation(dim)[:2]
                        newpos[i, params[0]] = X[i, params[0]] + (
                            X[int(RJ), params[0]] - X[i, params[1]]
                        ) * (r1 + 1) * np.sin(r2 * 360)
                        newpos[i, params[1]] = X[i, params[1]] + (
                            X[int(RJ), params[0]] - X[i, params[1]]
                        ) * (r1 + 1) * np.cos(r2 * 360)
                    else:
                        params = np.random.permutation(dim)
                        for j in range(1, int(np.floor(dim / 2)) + 1):
                            newpos[i, 2 * j - 2] = X[i, params[2 * j - 2]] + (
                                X[int(RJ), params[0]] - X[i, params[2 * j - 2]]
                            ) * (r1 + 1) * np.sin(r2 * 360)
                            newpos[i, 2 * j - 1] = X[i, params[2 * j - 1]] + (
                                X[int(RJ), params[0]] - X[i, params[2 * j - 1]]
                            ) * (r1 + 1) * np.cos(r2 * 360)
                else:  # exploitation phase
                    r3 = np.random.rand()
                    r4 = np.random.rand()
                    C1 = 2 * r4 * (1 - t / self.max_Iter)
                    RJ = int(
                        np.ceil((N - 1) * np.random.rand())
                    )  # Roulette Wheel Selection
                    while RJ == i:
                        RJ = int(np.ceil((N - 1) * np.random.rand()))
                    alpha = 3 / 2
                    sigma = (
                        gamma(1 + alpha)
                        * np.sin(np.pi * alpha / 2)
                        / (gamma((1 + alpha) / 2) * alpha * 2 ** ((alpha - 1) / 2))
                    ) ** (
                        1 / alpha
                    )  # Levy flight
                    u = np.random.randn(dim) * sigma
                    v = np.random.randn(dim)
                    S = u / np.abs(v) ** (1 / alpha)
                    KD = 0.05
                    LevyFlight = KD * S
                    newpos[i, :] = (
                        r3 * Xgb
                        - r4 * X[i, :]
                        + C1 * LevyFlight * (X[int(RJ), :] - X[i, :])
                    )
                # boundary
                Flag4ub = newpos[i, :] > ub
                Flag4lb = newpos[i, :] < lb
                newpos[i, :] = (
                    (newpos[i, :] * (~(Flag4ub + Flag4lb)))
                    + ub * Flag4ub
                    + lb * Flag4lb
                )
                # newfit[i, 0] = fun(newpos[i, :])  # fitness calculation
                newfit[i, 0] = self.loss_func(
                    x_train[:, newpos[i, :] > self.thres],
                    x_test[:, newpos[i, :] > self.thres],
                    y_train,
                    y_test,
                )
                Counts_run = Counts_run + 1

                if newfit[i, 0] < fit[i, 0]:
                    X[i, :] = newpos[i, :]
                    fit[i, 0] = newfit[i, 0]

            for i in range(N):
                # whale falls
                if kk[i] <= WF:
                    RJ = int(np.ceil((N - 1) * np.random.rand()))
                    r5, r6, r7 = np.random.rand(3)
                    C2 = 2 * N * WF
                    stepsize2 = r7 * (ub - lb) * np.exp(-C2 * t / self.max_Iter)
                    newpos[i, :] = (r5 * X[i, :] - r6 * X[RJ, :]) + stepsize2
                    # boundary
                    Flag4ub = newpos[i, :] > ub
                    Flag4lb = newpos[i, :] < lb
                    newpos[i, :] = (
                        (newpos[i, :] * (~(Flag4ub + Flag4lb)))
                        + ub * Flag4ub
                        + lb * Flag4lb
                    )
                    # newfit[i,0] = fun(newpos[i,:])  # fitness calculation
                    newfit[i, 0] = self.loss_func(
                        x_train[:, newpos[i, :] > self.thres],
                        x_test[:, newpos[i, :] > self.thres],
                        y_train,
                        y_test,
                    )
                    Counts_run += 1
                    if newfit[i, 0] < fit[i, 0]:
                        X[i, :] = newpos[i, :]
                        fit[i, 0] = newfit[i, 0]

            fval, index = np.min(fit), np.argmin(fit)
            if fval < fitG:
                fitG = fval
                Xgb = X[index, :]
            curve[t - 1] = fitG
            print(f"\nIteration {t} Best (BWO)= {fitG}")
            t += 1
        Pos = np.arange(dim) + 1
        Sf = Pos[(Xgb > self.thres) == 1] - 1

        BWO = {}
        BWO["sf"] = Sf
        BWO["c"] = curve
        return BWO
