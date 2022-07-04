# -*- coding: utf-8 -*-
import time
import gurobipy as gp
from helper_functions import GenShifts, GenMatrix


def nursing_BendersDecomposition(SimEngine, Inits, Heuristic,
        H, L, ShiftLengths, MinWorkers, MaxWorkers, MaxHours, YFix):
    
    # Small number
    EPS = 0.0001
    
    # Time periods
    T = range(H)
    
    # Set of scenarios
    S = range(len(SimEngine.Scenario))
    
    # Set of worker levels
    N = range(MinWorkers, MaxWorkers + 1)
    
    # Set of shifts
    Shifts = GenShifts(ShiftLengths, H, L)

    # Set of shifts
    G = range(len(Shifts))
    
    # Shift-time matrix
    A = GenMatrix(Shifts, L, T)

    # --------------------
    # --- Gurobi model ---
    # --------------------
    Model = gp.Model()
    Model._InitialCutsTime = 0
    Model._InitialCutsAdded = 0
    Model._HeuristicTime = 0
    Model._HeuristicValue = None

    # X[g] is the number of care workers rostered for shift type g
    X = {g: Model.addVar(vtype=gp.GRB.INTEGER) for g in G}

    # Y[t] is the number of care workers working time period t
    Y = {t: Model.addVar(
        vtype=gp.GRB.INTEGER, lb=MinWorkers, ub=MaxWorkers) for t in T}
    
    # Fix solution if given
    if YFix is not None:
        for t in YFix:
            Model.addConstr(Y[t] == YFix[t])

    # Z[xi, t] = 1 if and only if Y[t] = xi
    Z = {(xi, t): Model.addVar(vtype=gp.GRB.BINARY) for xi in N for t in T}

    # Theta[s, t] estimates the delay of requests released in time period t
    Theta = {(s, t): Model.addVar() for s in S for t in T}

    # Scheduling constraints
    for t in T:

        # Correct care workers available given the shift schedule X
        Model.addConstr(Y[t] == gp.quicksum(A[t][g] * X[g] for g in G))

    # Maximum working hours
    Model.addConstr(
        gp.quicksum(L * Y[t] for t in T) <= 60 * MaxHours)

    # Link Z Y
    for t in T:
        Model.addConstr(gp.quicksum(Z[xi, t] for xi in N) == 1)
        Model.addConstr(gp.quicksum(xi * Z[xi, t] for xi in N) == Y[t])

    # Objective function
    Model.setObjective(gp.quicksum(
        Theta[_] for _ in Theta) / len(S), gp.GRB.MINIMIZE)
    
    # --------------------
    # --- Initial Cuts ---
    # --------------------
    if Inits:
        InitialCutsStartTime = time.time()
        for s in S:

            # W[t, xi_1] is the delay for time period
            #  t with xi_1 workers in time period t
            W1 = {}

            # W2[t, xi_2, xi_1] is the delay time for time period t
            # with xi_1 workers in time period t and xi_2 in t - 1
            W2 = {}

            for t in T:
                for xi_1 in N:
                    # One dimensional information
                    Delay = SimEngine.Simulate(s, [(t, xi_1)])
                    W1[t, xi_1] = Delay[t]
                    # Initial values for W2
                    for xi_2 in N:
                        W2[t, xi_2, xi_1] = Delay[t]
                    # Fix the W2 values
                    if t > 0 and xi_1 < N[-1]:
                        for xi_2 in N[:-1]:
                            Delay = SimEngine.Simulate(s, [(t - 1, xi_2),
                                                    (t, xi_1)])
                            W2[t, xi_2, xi_1] = Delay[t]
                            if Delay[t] <= W1[t, xi_1] + EPS:
                                break

                    # We can stop fixing W2 if there is no delay
                    elif Delay[t] == 0:
                        break

                # Add cuts
                for xi_1 in N:
                    if t > 0:
                        
                        # First initial cut
                        Model._InitialCutsAdded += 1
                        Model.addConstr(Theta[s, t] >=
                                    gp.quicksum(Z[xi_2, t-1] *
                                                W2[t, xi_2, xi_1]
                                                for xi_2 in N) -
                                    gp.quicksum(max(W2[t, xi_2, xi_1] -
                                                    W2[t, xi_2, xi_p]
                                                    for xi_2 in N) *
                                                Z[xi_p, t]
                                                for xi_p in
                                                N[N.index(xi_1)+1:]))
                for xi_2 in N:
                    if t > 0:
                        
                        # Second initial cut
                        Model._InitialCutsAdded += 1
                        Model.addConstr(Theta[s, t] >=
                                    gp.quicksum(Z[xi_1, t] *
                                                W2[t, xi_2, xi_1]
                                                for xi_1 in N) -
                                    gp.quicksum(max(W2[t, xi_2, xi_1] -
                                                    W2[t, xi_p, xi_1]
                                                    for xi_1 in N) *
                                                Z[xi_p, t-1]
                                                for xi_p in
                                                N[N.index(xi_2)+1:]))
        # Initial cuts time
        Model._InitialCutsTime += time.time() - InitialCutsStartTime

    # -----------------
    # --- Heuristic ---
    # -----------------
    HeuristicValue = None
    if Heuristic:
        HeuristicStartTime = time.time()

        # Get the total duration of requests which arrive
        Durations = {t: 0 for t in T}
        Durations[len(T)] = 0
        for s in S:
            for k in SimEngine.Scenario[s]:
                Durations[round(k[0] // L)] += k[1]

        # Curve[d][t] is the average number of service hours in
        # time period t of day d
        Curve = [Durations[t] / (len(S) * L) + 2 for t in T]

        # MIP model for the heuristic
        Heuristic = gp.Model()

        # Variables: equivalent to main model
        XH = {g: Heuristic.addVar(vtype=gp.GRB.INTEGER) for g in G}

        YH = {t: Heuristic.addVar(
            vtype=gp.GRB.INTEGER, lb=MinWorkers, ub=MaxWorkers) for t in T}

        # Dist[t, d] is the absolute distance between Curve[t] and Y[t]
        Dist = {t: Heuristic.addVar() for t in T}

        # Constraints
        for t in T:
            # Scheduling constraint
            Heuristic.addConstr(
                YH[t] == gp.quicksum(A[t][g] * XH[g] for g in G))

            # Absolute value
            Heuristic.addConstr(Dist[t] >= Curve[t] - YH[t])
            Heuristic.addConstr(Dist[t] >= YH[t] - Curve[t])

        # Maxium working hours
        Heuristic.addConstr(gp.quicksum(L * YH[t] for t in T) <= 60 * MaxHours)

        # Heuristic objective: minimize difference between
        # total agent hours and hourly workloads
        Heuristic.setObjective(gp.quicksum(Dist[t] for t in T) / len(T))
        
        # Solve heuristic
        Heuristic.setParam("OutputFlag", 0)
        Heuristic.optimize()

        # Retrieve the solution and give it to Gurobi
        HeuristicValue = 0
        for g in G:
            X[g].Start = round(XH[g].x)
        for t in T:
            Y[t].Start = round(YH[t].x)
            for xi in N:
                if round(YH[t].x) == xi:
                    Z[xi, t].Start = 1
                else:
                    Z[xi, t].Start = 0

        # Simulate the heuristic solution
        for s in S:
            ThetaH = SimEngine.Simulate(s, [(t, round(YH[t].x)) for t in T])
            
            # Set starting Theta[s, t]
            for t in T:
                HeuristicValue += ThetaH[t]
                Theta[s, t].Start = ThetaH[t]

        Model._HeuristicValue = HeuristicValue / len(S)
        Model._HeuristicTime += time.time() - HeuristicStartTime
    
    # ----------------
    # --- Optimize ---
    # ----------------
    Model._TimeInCallback = 0
    Model._BestObj = gp.GRB.INFINITY
    Model._BestY = None
    Model._BestZ = None
    Model._BestTheta = None
    Model._CutsAdded = 0
    
    # Put the variables into lists
    YVar = list(Y.values())
    ZVar = list(Z.values())
    ThetaVar = list(Theta.values())
    

    # Callback function
    def Callback(model, where):
        CallbackStartTime = time.time()
        if where == gp.GRB.Callback.MIPSOL:
    
            # Retrieve the current solution
            YV = model.cbGetSolution(Y)
            ThetaV = model.cbGetSolution(Theta)
            SumDelay = 0
    
            # Simulate each scenario
            for s in S:

                # Get the care worker levels of today
                Levels = [round(YV[t]) for t in T]

                Delay = SimEngine.Simulate(s, [(t, Levels[t]) for t in T])
                SumDelay += sum(Delay[t] for t in T)

                # Add cuts
                for t in T:
                    
                    # No cut if Theta is high enough
                    if ThetaV[s, t] >= Delay[t] - EPS:
                        continue

                    # Update Theta and log the cut
                    ThetaV[s, t] = Delay[t]
                    Model._CutsAdded += 1

                    # Get the starting neighborhood
                    tLow = t
                    tHigh = t

                    # Expand the neighborhood while legal
                    while True:
                        tLow = max(tLow - 1, 0)
                        tHigh = min(tHigh + 1, T[-1])

                        # If the neighborhood is good enough then stop
                        if SimEngine.Simulate(s, [(tt, Levels[tt]) for tt in
                                           range(tLow, tHigh + 1)])[t] >= \
                                Delay[t] - EPS:
                            break

                    # Contract from the right if we can
                    while tHigh > t and \
                        SimEngine.Simulate(s, [(tt, Levels[tt]) for tt in
                                        range(tLow, tHigh - 1)])[t] >= \
                            Delay[t] - EPS:
                        tHigh -= 1

                    # Contract from the left if we can
                    while tLow < t and \
                        SimEngine.Simulate(s, [(tt, Levels[tt]) for tt in
                                        range(tLow + 1, tHigh)])[t] >= \
                            Delay[t] - EPS:
                        tLow += 1

                    # Base is a valid bound if we don't increase Y[t]
                    Base = SimEngine.Simulate(s, [(t, Levels[t])])[t]

                    # If we increase workers at t
                    Term1 = gp.quicksum((Delay[t] -
                                         SimEngine.Simulate(s, [(t, xi)])[t]) *
                                        Z[xi, t] for xi in
                                        N[N.index(Levels[t]) + 1:])

                    # If we dont increase workers at t
                    Term2 = (Delay[t] - Base) * (gp.quicksum(
                        Z[xi, tt] for tt in range(tLow, tHigh + 1)
                        for xi in N[N.index(Levels[tt]) + 1:] if tt != t))

                    # Add the Benders cut as a lazy constraint
                    Model._CutsAdded += 1
                    model.cbLazy(Theta[s, t] >= Delay[t] - Term1 - Term2)

    
            # Primal heuristic
            if SumDelay < Model._BestObj:
    
                # Store the better solution
                Model._BestObj = SumDelay
                Model._BestY = model.cbGetSolution(YVar)
                Model._BestZ = model.cbGetSolution(ZVar)
                Model._BestTheta = [ThetaV[k] for k in Theta]
    
        # Pass the better solution to Gurobi
        if where == gp.GRB.callback.MIPNODE and \
            model.cbGet(gp.GRB.callback.MIPNODE_STATUS) == gp.GRB.OPTIMAL and \
                model.cbGet(gp.GRB.callback.MIPNODE_OBJBST) > Model._BestObj + EPS:
    
            # Pass the better solution to Gurobi
            Model.cbSetSolution(YVar, Model._BestY)
            Model.cbSetSolution(ZVar, Model._BestZ)
            Model.cbSetSolution(ThetaVar, Model._BestTheta)
    
        Model._TimeInCallback += time.time() - CallbackStartTime    
        
    # Solve
    Model.setParam("LazyConstraints", 1)
    # Model.setParam("OutputFlag", 0)
    Model.optimize(Callback)
    Model._TotalTime = Model.Runtime + Model._InitialCutsTime \
        + Model._HeuristicTime

    
    
    return Model, {t: round(Y[t].x) for t in Y}
        

    
    