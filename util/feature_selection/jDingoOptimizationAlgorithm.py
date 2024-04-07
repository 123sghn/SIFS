import numpy as np
from losses.jFitnessFunction import jFitnessFunction


class jDingoOptimizationAlgorithm:
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
        P=0.5,
        Q=0.7,
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
        self.P = P
        self.Q = Q

    def optimize(self, x_train, x_test, y_train, y_test):

        dim = x_train.shape[1]
        lb = 0
        ub = 1
        N = self.N

        P = (
            self.P
        )  # Hunting or Scavenger?  rate.  See section 3.0.4, P and Q parameters analysis
        Q = self.Q  # Group attack or persecution?

        beta1 = -2 + 4 * np.random.random()  # -2 < beta < 2     Used in Eq. 2,
        beta2 = -1 + 2 * np.random.random()  # -1 < beta2 < 1    Used in Eq. 2,3, and 4
        naIni = 2  # minimum number of dingoes that will attack
        naEnd = N / naIni  # maximum number of dingoes that will attack
        na = int(
            naIni + (naEnd - naIni) * np.random.random()
        )  # number of dingoes that will attack, used in Attack.m Section 2.2.1: Group attack
        X = np.random.random((N, dim)) * (ub - lb) + lb
        curve = np.zeros(self.max_Iter)
        Fitness = np.zeros(N)
        v = np.zeros_like(X)

        for i in range(X.shape[0]):
            Fitness[i] = self.loss_func(
                x_train[:, X[i, :] > self.thres],
                x_test[:, X[i, :] > self.thres],
                y_train,
                y_test,
            )  # get fitness
        minIdx = np.argmin(
            Fitness
        )  # the min fitness value fitG and the position minIdx
        Xgb = X[minIdx, :]  # the best vector
        maxIdx = np.argmax(
            Fitness
        )  # the max fitness value vMax and the position maxIdx
        curve[0] = Fitness[minIdx]

        survival = self._survival_rate(
            Fitness, Fitness[minIdx], Fitness[maxIdx]
        )  # Section 2.2.4 Dingoes'survival rates
        t = 0  # Loop counter

        for t in range(self.max_Iter):
            for r in range(N):
                sumatory = 0
                if np.random.random() < P:  # If Hunting?
                    sumatory = self._attack(
                        N, na, X, r
                    )  # Section 2.2.1, Strategy 1: Part of Eq.2
                    if np.random.random() < Q:  # If group attack?
                        v[r, :] = beta1 * sumatory - Xgb  # Strategy 1: Eq.2
                    else:  # Persecution
                        r1 = int(np.round((N - 1) * np.random.random()))
                        v[r, :] = Xgb + beta1 * (np.exp(beta2)) * (
                            (X[r1, :] - X[r, :])
                        )  # Section 2.2.2, Strategy 2:  Eq.3
                else:  # Scavenger
                    r1 = int(np.round((N - 1) * np.random.random()))

                    v[r, :] = (
                        np.exp(beta2) * X[r1, :] - ((-1) ** self._getBinary()) * X[r, :]
                    ) / 2  # Section 2.2.3, Strategy 3: Eq.4

                if (
                    survival[r] <= 0.3
                ):  # Section 2.2.4, Algorithm 3 - Survival procedure
                    band = 1
                    while band:
                        r1 = int(np.round((N - 1) * np.random.random()))
                        r2 = int(np.round((N - 1) * np.random.random()))
                        if r1 != r2:
                            band = 0

                    v[r, :] = (
                        Xgb + (X[r1, :] - ((-1) ** self._getBinary()) * X[r2, :]) / 2
                    )  # Section 2.2.4, Strategy 4: Eq.6

                Flag4ub = v[r, :] > ub
                Flag4lb = v[r, :] < lb
                v[r, :] = (
                    (v[r, :] * ~(Flag4ub + Flag4lb)) + ub * Flag4ub + lb * Flag4lb
                )  # Return back the search agents that go beyond the boundaries of the search space.

                Fnew = self.loss_func(
                    x_train[:, v[r, :] > self.thres],
                    x_test[:, v[r, :] > self.thres],
                    y_train,
                    y_test,
                )  # Evaluate new solutions

                if Fnew <= Fitness[r]:  # Update if the solution improves
                    X[r, :] = v[r, :]
                    Fitness[r] = Fnew

                if Fnew <= curve[0]:
                    Xgb = v[r, :]
                    curve[0] = Fnew

            curve[t] = curve[0]
            print("\nIteration {} Best (DOA) = {}".format(t + 1, curve[0]))

            maxIdx = np.argmax(Fitness)
            survival = self._survival_rate(
                Fitness, curve[0], Fitness[maxIdx]
            )  # Section 2.2.4 Dingoes'survival rates

        Pos = np.arange(1, dim + 1)
        Sf = Pos[(Xgb > self.thres) == 1] - 1

        DOA = {}
        DOA["sf"] = Sf  # best_X
        DOA["c"] = curve

        return DOA

    @staticmethod
    def _survival_rate(fit, minimum, maximum):
        o = []
        for i in range(len(fit)):
            o.append((maximum - fit[i]) / (maximum - minimum))
        return o

    @staticmethod
    def _attack(SearchAgents_no, na, Positions, r):
        sumatory = 0
        vAttack = vectorAttack(SearchAgents_no, na)

        for j in range(len(vAttack)):
            sumatory += Positions[vAttack[j], :] - Positions[r, :]
        sumatory /= na
        return sumatory

    @staticmethod
    def _getBinary():
        if np.random.random() < 0.5:
            return 0
        else:
            return 1


def vectorAttack(SearchAgents_no, na):
    c = 1
    vAttack = []
    while c <= na:
        idx = int(np.round((SearchAgents_no - 1) * np.random.random()))
        if idx not in vAttack:

            vAttack.append(idx)
            c += 1
    return vAttack

    @staticmethod
    def _findrep(val, vector):
        # return 1= repeated  0= not repeated
        band = 0
        for i in range(len(vector)):
            if val == vector[i]:
                band = 1
                break
        return band
