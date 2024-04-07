from losses.jFitnessFunction import jFitnessFunction
import random
import numpy as np


class jBinaryDwarfMongooseOptimizer:
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
        nBabysitter=3,
        peep=1,
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
        self.nBabysitter = nBabysitter
        self.peep = peep

    def optimize(self, x_train, x_test, y_train, y_test):
        # Problem Definition
        dim = x_train.shape[1]
        lb = 0  # Decision Variables Lower Bound
        ub = 1  # Decision Variables Upper Bound

        # DMOA Settings
        nBabysitter = self.nBabysitter  # Number of babysitters
        nAlphaGroup = self.N - nBabysitter  # Number of Alpha group
        nScout = nAlphaGroup  # Number of Scouts
        L = round(0.6 * dim * nBabysitter)  # Babysitter Exchange Parameter
        peep = self.peep  # Alpha female's vocalization
        VarSize = [1, dim]

        # Empty Mongoose Structure
        class Mongoose:
            def __init__(self):
                self.Position = None
                self.Cost = None
                self.Acc = None

        # Initialize Population Array
        pop = np.empty(nAlphaGroup, dtype=object)

        # Initialize Best Solution Ever Found
        BestSol = Mongoose()
        BestSol.Cost = np.inf
        tau = np.inf
        Iter = 1
        sm = np.full(nAlphaGroup, np.inf)

        # Create Initial Population
        for i in range(nAlphaGroup):
            pop[i] = Mongoose()
            pop[i].Position = np.random.uniform(lb, ub, size=dim)
            pop[i].Cost = self.loss_func(
                x_train[:, pop[i].Position > self.thres],
                x_test[:, pop[i].Position > self.thres],
                y_train,
                y_test,
            )

            if pop[i].Cost < BestSol.Cost:
                BestSol = pop[i]

        # Abandonment Counter
        C = np.zeros(nAlphaGroup)
        CF = (1 - Iter / self.max_Iter) ** (2 * Iter / self.max_Iter)

        # Array to Hold Best Cost Values
        curve = np.zeros(self.max_Iter)

        ## DMOA Main Loop
        for it in range(self.max_Iter):
            # Alpha group
            F = np.zeros(nAlphaGroup)
            MeanCost = np.mean([p.Cost for p in pop])
            for i in range(nAlphaGroup):
                # Calculate Fitness Values and Selection of Alpha
                F[i] = np.exp(-pop[i].Cost / MeanCost)  # Convert Cost to Fitness
            P = F / np.sum(F)

            # Foraging led by Alpha female
            for m in range(nAlphaGroup):
                # Select Alpha female
                i = self._roulette_wheel_selection(P)

                # Choose k randomly, not equal to Alpha
                K = list(range(i)) + list(range(i + 1, nAlphaGroup))
                k = np.random.choice(K)

                # Define Vocalization Coeff.
                phi = (peep / 2) * np.random.uniform(-1, +1, size=dim)

                # New Mongoose Position
                newpop = Mongoose()
                newpop.Position = pop[i].Position + phi * (
                    pop[i].Position - pop[k].Position
                )

                # Check boundary lb, ub
                Flag_UB = (
                    newpop.Position > ub
                )  # check if they exceed (up) the boundaries
                Flag_LB = (
                    newpop.Position < lb
                )  # check if they exceed (down) the boundaries
                newpop.Position = (
                    (newpop.Position * (~(Flag_UB + Flag_LB)))
                    + ub * Flag_UB
                    + lb * Flag_LB
                )

                # Evaluation
                newpop.Cost = self.loss_func(
                    x_train[:, newpop.Position > self.thres],
                    x_test[:, newpop.Position > self.thres],
                    y_train,
                    y_test,
                )

                # Comparison
                if newpop.Cost <= pop[i].Cost:
                    pop[i] = newpop
                else:
                    C[i] += 1

            for i in range(nScout):
                # Choose k randomly, not equal to i
                K = list(range(1, i)) + list(range(i + 1, nAlphaGroup))
                k = K[np.random.randint(len(K))]

                # Define Vocalization Coeff.
                phi = (peep / 2) * np.random.uniform(-1, 1, VarSize)
                phi = phi.reshape(-1)
                # New Mongoose Position
                newpop.Position = pop[i].Position + phi * (
                    pop[i].Position - pop[k].Position
                )

                # Check boundary
                Flag_UB = (
                    newpop.Position > ub
                )  # check if they exceed (up) the boundaries
                Flag_LB = (
                    newpop.Position < lb
                )  # check if they exceed (down) the boundaries
                newpop.Position = (
                    (newpop.Position * (~(Flag_UB + Flag_LB)))
                    + ub * Flag_UB
                    + lb * Flag_LB
                )
                # Evaluation
                newpop.Cost = self.loss_func(
                    x_train[:, newpop.Position > self.thres],
                    x_test[:, newpop.Position > self.thres],
                    y_train,
                    y_test,
                )

                # Sleeping mould
                sm[i] = (newpop.Cost - pop[i].Cost) / max(newpop.Cost, pop[i].Cost)

                # Comparison
                if newpop.Cost <= pop[i].Cost:
                    pop[i] = newpop
                else:
                    C[i] += 1

            # Babysitters
            for i in range(nBabysitter):
                newtau = np.mean(sm)

                if C[i] >= L:
                    M = (pop[i].Position * sm) / pop[i].Position

                    if newtau < tau:
                        newpop.Position = pop[
                            i
                        ].Position - CF * phi * np.random.rand() * (pop[i].Position - M)
                    else:
                        newpop.Position = pop[
                            i
                        ].Position + CF * phi * np.random.rand() * (pop[i].Position - M)

                    tau = newtau
                    Flag_UB = (
                        newpop.Position > ub
                    )  # check if they exceed (up) the boundaries
                    Flag_LB = (
                        newpop.Position < lb
                    )  # check if they exceed (down) the boundaries
                    newpop.Position = (
                        (newpop.Position * (~(Flag_UB + Flag_LB)))
                        + ub * Flag_UB
                        + lb * Flag_LB
                    )

                    C[i] = 0

            # Update Best Solution Ever Found
            for i in range(nAlphaGroup):
                if pop[i].Cost <= BestSol.Cost:
                    BestSol = pop[i]

            # Store Best Cost Ever Found
            curve[it] = BestSol.Cost

            # Display Iteration Information
            print("Iteration {}: Best Cost = {}".format(it, curve[it]))
        Pos = np.arange(1, dim + 1)
        Sf = Pos[(BestSol.Position > 0.5) == 1]

        # 主循环结束
        BDM = {}
        BDM["sf"] = Sf  # best_X
        BDM["c"] = curve
        return BDM

    @staticmethod
    def _roulette_wheel_selection(P):
        r = random.uniform(0, 1)
        s = np.sum(P)
        P = P / s
        C = np.cumsum(P)
        j = np.where(r <= C)[0][0]
        return j
