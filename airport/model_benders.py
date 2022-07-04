# -*- coding: utf-8 -*-
import time
import gurobipy as gp
import heapq


def airport_BendersDecomposition(SimEngine, Inits, H, L, Start, Arrive, 
                                 MaxDesks, DeskCost, QueueCost):
    

    # Small number
    EPS = 0.0001

    # Set of time periods
    T = range(H)
    
    # Set of scenarios
    S = range(len(SimEngine.Scenarios))
    
    # Set of counter levels
    N = range(MaxDesks + 1)

    # --------------------
    # --- Gurobi model ---
    # --------------------
    Model = gp.Model()
    Model._InitialCutsTime = 0
    Model._InitialCutsAdded = 0

    # Y[t] is the number of check-in counters open at time t
    Y = {t: Model.addVar(ub=MaxDesks, vtype=gp.GRB.INTEGER) for t in T}

    # Z[xi, t] = 1 if and only if Y[t] = xi
    Z = {(xi, t): Model.addVar(vtype=gp.GRB.BINARY) for xi in N for t in T}

    # Theta[s, t] estiamtes the waiting time of passengers
    Theta = {(s, t): Model.addVar() for s in S for t in T}

    # Constraints
    for t in T:

        # Link the Y and Z variables
        Model.addConstr(gp.quicksum(Z[xi, t] for xi in N) == 1)
        Model.addConstr(gp.quicksum(Z[xi, t] * xi for xi in N) == Y[t])

        # Total processing time of some passengers about t
        After = max(sum(k[2] for k in SimEngine.Scenarios[s] if k[0] >= t*L) for s in S)
        Before = max(sum(k[2] for k in SimEngine.Scenarios[s]
                         if Start[k[1]] + len(Arrive) <= t) for s in S)

        # Open enough counters to clear the passengers
        Model.addConstr(L * gp.quicksum(Y[tt] for tt in T[t:]) >= After)
        Model.addConstr(L * gp.quicksum(Y[tt] for tt in T[:t + 1]) >= Before)

    # Objective function
    Model.setObjective(DeskCost * gp.quicksum(Y[t] for t in T) +
                   QueueCost * gp.quicksum(Theta[s, t]/(L * len(S))
                                           for s in S for t in T))

        
    # --------------------
    # --- Initial Cuts ---
    # --------------------
    if Inits:
        InitialCutsStartTime = time.time()
        print("# Initial cuts:", end=" ")
        for s in S:
            print(s, end=" ")

            # W[t, xi_1] is the queuing for time period t with xi_1
            # counters open in time period t
            W1 = {}

            # W2[t, xi_2, xi_1] is the queuing time for time period t
            # with xi_1 counters in time period t and xi_2 in t - 1
            W2 = {}

            for t in T:
                for xi_1 in N:

                    # One dimensional information
                    QueuingTime = SimEngine.Simulate(s, [(t, xi_1)])
                    W1[t, xi_1] = QueuingTime[t]

                    # Initial values for W2
                    for xi_2 in N:
                        W2[t, xi_2, xi_1] = QueuingTime[t]

                    if t > 0 and xi_1 < N[-1]:
                        for xi_2 in N[:-1]:

                            # Fix the 2D queuing times
                            QueuingTime = SimEngine.Simulate(s, [(t-1, xi_2), (t, xi_1)])
                            W2[t, xi_2, xi_1] = QueuingTime[t]
                            if QueuingTime[t] <= W1[t, xi_1] + EPS:
                                break

                    # We can stop fixing W2 if there is no delay
                    elif QueuingTime[t] == 0:
                        break

                # Add cuts
                for xi_1 in N:
                    if t > 0:

                        # First initial cut
                        Model._InitialCutsAdded += 1
                        Model.addConstr(Theta[s, t] >=
                                    gp.quicksum(Z[xi_2, t-1]*W2[t, xi_2, xi_1]
                                                for xi_2 in N) -
                                    gp.quicksum(max(W2[t, xi_2, xi_1] -
                                                    W2[t, xi_2, xi_p]
                                                    for xi_2 in N)*Z[xi_p, t]
                                                for xi_p in N[xi_1+1:]))
                for xi_2 in N:
                    if t > 0:

                        # Second initial cut
                        Model._InitialCutsAdded += 1
                        Model.addConstr(Theta[s, t] >=
                                    gp.quicksum(Z[xi_1, t]*W2[t, xi_2, xi_1]
                                                for xi_1 in N) -
                                    gp.quicksum(max(W2[t, xi_2, xi_1] -
                                                    W2[t, xi_p, xi_1]
                                                    for xi_1 in N)*Z[xi_p, t-1]
                                                for xi_p in N[xi_2+1:]))

        # Time spent generating initial cuts
        Model._InitialCutsTime += time.time() - InitialCutsStartTime
        print("Done \n")

    # ----------------
    # --- Callback ---
    # ----------------
    Model._TimeInCallback = 0
    Model._BestObj = gp.GRB.INFINITY
    Model._CutsAdded = 0
    Model._CutSizes = {t + 1: 0 for t in T}

    # Put the variables into lists
    YVar = list(Y.values())
    ZVar = list(Z.values())
    ThetaVar = list(Theta.values())

    def Callback(model, where):
        CallbackStartTime = time.time()

        # Integer solutions
        if where == gp.GRB.Callback.MIPSOL:

            # Retrieve the current solution
            YVal = model.cbGetSolution(Y)
            ThetaVal = model.cbGetSolution(Theta)

            # Get the current levels
            Levels = [round(YVal[t]) for t in T]

            # Total performance of the current solution
            SumDelay = 0

            # Cuts on s
            for s in S:

                # Simulate the current solution; note total performance
                QueueTime = SimEngine.Simulate(s, [(t, Levels[t]) for t in T])
                SumDelay += sum(QueueTime[t] for t in T)

                # Cuts on t
                for t in T:

                    # If Theta[s, t] is correct then move on
                    if ThetaVal[s, t] >= QueueTime[t] - EPS:
                        continue

                    # Updata Thetas
                    ThetaVal[s, t] = QueueTime[t]

                    # --------------------
                    # --- Neighborhood ---
                    # --------------------
                    tLow = t
                    tHigh = t

                    # Two-sided window expansion
                    while True:
                        tLow = max(tLow - 1, 0)
                        tHigh = min(tHigh + 1, T[-1])
                        if SimEngine.Simulate(s, [(tt, Levels[tt]) for tt in range(
                                tLow, tHigh + 1)])[t] >= QueueTime[t] - EPS:
                            break

                    # Contract from the right if we can
                    while tHigh > t and SimEngine.Simulate(
                            s, [(tt, Levels[tt]) for tt in range(
                                tLow, tHigh - 1)])[t] >= QueueTime[t] - EPS:
                        tHigh -= 1

                    # Contract from the left if we can
                    while tLow < t and SimEngine.Simulate(
                            s, [(tt, Levels[tt]) for tt in range(
                                tLow + 1, tHigh)])[t] >= QueueTime[t] - EPS:
                        tLow += 1


                    # Base is a valid  bound if we don't increase y[k]
                    Base = SimEngine.Simulate(s, [(t, Levels[t])])[t]

                    # Increase levels at t
                    Term1 = gp.quicksum((QueueTime[t] - SimEngine.Simulate(
                        s, [(t, xi)])[t]) * Z[xi, t] for xi in N[N.index(Levels[t]) + 1:])

                    # Increase levels at others
                    Term2 = (QueueTime[t] - Base) * (gp.quicksum(
                        Z[xi, tt] for tt in range(tLow, tHigh + 1)
                        for xi in N[N.index(Levels[tt]) + 1:] if tt != t))

                    # Add the cut
                    Model._CutsAdded += 1
                    Model._CutSizes[round(tHigh - tLow + 1)] += 1
                    model.cbLazy(Theta[s, t] >= QueueTime[t] - Term1 - Term2)

            # Primal heuristic
            CurrentObj = sum(Levels)*DeskCost + SumDelay*QueueCost/(L * len(S))
            if CurrentObj < Model._BestObj:

                # Store the better solution
                Model._BestObj = CurrentObj
                Model._BestY = model.cbGetSolution(YVar)
                Model._BestZ = model.cbGetSolution(ZVar)
                Model._BestTheta = [ThetaVal[k] for k in Theta]

        # Pass the better solution to Gurobi
        if where == gp.GRB.callback.MIPNODE and \
            model.cbGet(gp.GRB.callback.MIPNODE_STATUS) == gp.GRB.OPTIMAL and \
                model.cbGet(gp.GRB.callback.MIPNODE_OBJBST) > Model._BestObj + EPS:

            # Pass the better solution to Gurobi
            Model.cbSetSolution(YVar, Model._BestY)
            Model.cbSetSolution(ZVar, Model._BestZ)
            Model.cbSetSolution(ThetaVar, Model._BestTheta)
    
        Model._TimeInCallback += time.time() - CallbackStartTime

    # ---------------
    # --- Results ---
    # ---------------
    Model.Params.LazyConstraints = 1
    Model.Params.MIPGap = 0
    Model.optimize(Callback)
    Model._TotalTime = Model.Runtime + Model._InitialCutsTime
    
    return Model, {t: round(Y[t].x) for t in T}


