# -*- coding: utf-8 -*-
import csv
import heapq
import numpy as np
import time
import gurobipy as gp
from gurobipy import GRB


def Optimize(Data, Scenario, Inits, Heuristic,
             ResultsFile, SolutionFile, SizesFile):

    # Small number for controlling floating point error
    EPS = 0.0001

    # Set of scenarios
    S = range(len(Scenario))

    # Set of days
    Days = range(len(Scenario[0]))

    # Set of time periods
    T = Data.T

    # Length of a time period
    L = Data.L

    # Minimum workers at one time
    MinWorkers = Data.MinWorkers

    # Max workers at one time
    MaxWorkers = Data.MaxWorkers

    # Set of worker levels
    N = range(MinWorkers, MaxWorkers + 1)

    # Endpoints of shifts
    Shifts = Data.Shifts

    # Set of shifts
    G = range(len(Shifts))

    # Shift-time matrix
    A = Data.A

    # --------------------
    # --- Gurobi model ---
    # --------------------
    print("Creating Gurobi model")
    m = gp.Model()

    # X[g, d] is the number of care workers working shift g on day d
    X = {(g, d): m.addVar(vtype=GRB.INTEGER)
         for g in G for d in Days}

    # Y[t, d] is the number of care workers working in time period t on day d
    Y = {(t, d): m.addVar(vtype=GRB.INTEGER, lb=MinWorkers, ub=MaxWorkers)
         for t in T for d in Days}

    # Z[xi, t, d] = 1 if and only if Y[t, d] = xi
    Z = {(xi, t, d): m.addVar(vtype=GRB.BINARY)
         for xi in N for t in T for d in Days}

    # Theta[s, t, d] estimates the delay of the requests which arrive in
    # time period t of scenario s of day d
    Theta = {(s, t, d): m.addVar()
             for s in S for t in T for d in Days}

    # Scheduling constraints
    for d in Days:
        for t in T:

            # Relate the care worker variables to the shift schedule
            m.addConstr(Y[t, d] == gp.quicksum(A[t][g] * X[g, d] for g in G))

    # Impose maximum working hours
    m.addConstr(gp.quicksum(L * Y[t, d]
                            for t in T for d in Days) <= 60 * Data.MaxHours)

    # Make sure Z variables agree with the Y variables
    for d in Days:
        for t in T:

            # Precisely one of each Z variable is non-zero
            m.addConstr(gp.quicksum(Z[xi, t, d] for xi in N) == 1)

            # The correct Z variables are non-zero
            m.addConstr(gp.quicksum(xi * Z[xi, t, d] for xi in N) == Y[t, d])

    # Objective: sample-average approximation of the total delay of all jobs
    m.setObjective(gp.quicksum(Theta[s, t, d] for s in S for t in T
                               for d in Days) / len(S), GRB.MINIMIZE)

    # -------------------
    # --- Timekeeping ---
    # -------------------
    TotalTimeStart = time.time()
    HeuristicTime = 0
    InitialCutsTime = 0
    m._SimulationTime = 0
    m._TimeInCallback = 0

    # ---------------------------------
    # --- Discrete-event Simulation ---
    # ---------------------------------
    Simulations = {}
    CareStart = {}
    ReqsStart = {}

    # Simulate day d of scenario s using the levels specified in LevelsList
    def Simulate(s, d, LevelsList):
        SimStartTime = time.time()

        # Set up the worker levels
        Levels = [MaxWorkers for t in T]
        init = False
        if not LevelsList:
            init = True
        else:
            for (t, l) in LevelsList:
                Levels[t] = l

        # Check the cache
        TupleLevels = tuple(Levels)
        if (s, d, TupleLevels) in Simulations:
            return Simulations[s, d, TupleLevels]

        # Performance measures
        Delay = {t: 0 for t in T}

        # Stopping criteria
        if init:
            tEnd = T[-1] + 1
        else:
            tEnd = LevelsList[-1][0]+1

        # Set up the simulation
        if init or LevelsList[0][0] == 0:
            tUpto = 0
            Agents = []
            for i in range(Levels[0]):
                heapq.heappush(Agents, 0)
            Requests = Scenario[s][d]

        # Get the starting state of the simulation if the first time period
        else:  # with the maximum number of care workers is not time period 0
            tUpto = LevelsList[0][0]
            Agents = []
            for a in list(CareStart[s, d, tUpto]):
                heapq.heappush(Agents, a)
            for i in range(int(Levels[tUpto]), MaxWorkers):
                heapq.heappop(Agents)
            Requests = Scenario[s][d][ReqsStart[s, d, tUpto]:]

        # For when we get to the nightshift
        NightShift = False

        # Service the ith request
        for (i, k) in enumerate(Requests):

            # Stopping condition
            if k[0] > tEnd * L:
                break

            # Initialization
            if init:
                # If the next request doesn't arrive until the next
                if k[0] >= (tUpto + 1) * L:  # time period, then move on
                    tUpto += 1
                    # Populate the data structures
                    CareStart[s, d, tUpto] = tuple(Agents)
                    ReqsStart[s, d, tUpto] = i

            else:
                # If the next time period isn't the night shift
                if round(tUpto + 1) in T:

                    # If there are no workers available, or the next
                    #  request hasen'tarrived yet, then move on
                    while round(tUpto + 1) in T and \
                            max(k[0], Agents[0]) >= (tUpto + 1) * L:

                        # Add care workers
                        if Levels[tUpto + 1] > Levels[tUpto]:
                            for i in range(Levels[tUpto], Levels[tUpto + 1]):
                                heapq.heappush(Agents, L*(tUpto + 1))
                        # Remove care workers
                        if Levels[tUpto + 1] < Levels[tUpto]:
                            for i in range(Levels[tUpto + 1], Levels[tUpto]):
                                heapq.heappop(Agents)
                        # Move on
                        tUpto += 1

                # Go to the night shift once the next job must
                # start during the night shift
                elif not NightShift:
                    if int(max(k[0], min(Agents)) // L) not in T:
                        NightShift = True
                        # Drop down to night shift levels
                        for _ in range(len(Agents) - MinWorkers):
                            heapq.heappop(Agents)

            # Get the next care worker off the queue
            Agent = heapq.heappop(Agents)
            # Compute the delay
            ThisDelay = max(0, Agent - k[0])
            # Update the objective value
            Delay[int(k[0] // L)] += ThisDelay
            # Get the next idle time of the care worker
            Agent = max(Agent, k[0]) + k[1]
            # Put the care worker back into the agent queue
            heapq.heappush(Agents, Agent)

        # Dont bother handling any requests in the last
        # time period if we are in the initialization step
        if init:
            for tUpto in T[tUpto + 1:]:
                CareStart[s, d, tUpto] = tuple(Agents)
                ReqsStart[s, d, tUpto] = len(Scenario[s])

        # Cache and return the performance
        Simulations[s, d, TupleLevels] = Delay
        m._SimulationTime += time.time() - SimStartTime
        return Delay

    # Run the initialization
    for d in Days:
        for s in S:
            Simulate(s, d, None)

    # --------------------
    # --- Initial Cuts ---
    # --------------------
    InitialCutsAdded = 0
    if Inits:
        print("Generating initial cuts")
        print("Scenario:", end=" ")
        InitialCutsStartTime = time.time()
        for s in S:
            print(s, end=" ")
            for d in Days:

                # W[t, xi_1] is the delay for time period
                #  t with xi_1 workers in time period t
                W1 = {}

                # W2[t, xi_2, xi_1] is the delay time for time period t
                # with xi_1 workers in time period t and xi_2 in t - 1
                W2 = {}

                for t in T:
                    for xi_1 in N:
                        # One dimensional information
                        Delay = Simulate(s, d, [(t, xi_1)])
                        W1[t, xi_1] = Delay[t]
                        # Initial values for W2
                        for xi_2 in N:
                            W2[t, xi_2, xi_1] = Delay[t]
                        # Fix the W2 values
                        if t > 0 and xi_1 < N[-1]:
                            for xi_2 in N[:-1]:
                                Delay = Simulate(s, d, [(t - 1, xi_2),
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
                            InitialCutsAdded += 1
                            # First initial cut
                            m.addConstr(Theta[s, t, d] >=
                                        gp.quicksum(Z[xi_2, t-1, d] *
                                                    W2[t, xi_2, xi_1]
                                                    for xi_2 in N) -
                                        gp.quicksum(max(W2[t, xi_2, xi_1] -
                                                        W2[t, xi_2, xi_p]
                                                        for xi_2 in N) *
                                                    Z[xi_p, t, d]
                                                    for xi_p in
                                                    N[N.index(xi_1)+1:]))
                    for xi_2 in N:
                        if t > 0:
                            InitialCutsAdded += 1
                            # Second initial cut
                            m.addConstr(Theta[s, t, d] >=
                                        gp.quicksum(Z[xi_1, t, d] *
                                                    W2[t, xi_2, xi_1]
                                                    for xi_1 in N) -
                                        gp.quicksum(max(W2[t, xi_2, xi_1] -
                                                        W2[t, xi_p, xi_1]
                                                        for xi_1 in N) *
                                                    Z[xi_p, t-1, d]
                                                    for xi_p in
                                                    N[N.index(xi_2)+1:]))

        InitialCutsTime += time.time() - InitialCutsStartTime
        print(f"Spent {InitialCutsTime} generating initial cuts")

    # -----------------
    # --- Heuristic ---
    # -----------------
    HeuristicValue = None
    if Heuristic:
        print("Computing heuristic solution")
        HeuristicStartTime = time.time()

        # Get the total duration of requests which arrive
        Durations = {(t, d): 0 for t in T for d in Days}
        for d in Days:
            for s in S:
                for k in Scenario[s][d]:
                    Durations[round(k[0] // L), d] += k[1]

        # Curve[d][t] is the average number of service hours in
        # time period t of day d
        Curve = {d: [Durations[t, d] / (len(S) * L) + 2 for t in T]
                 for d in Days}

        # MIP model for the heuristic
        Heuristic = gp.Model()

        XH = {(g, d): Heuristic.addVar(vtype=GRB.INTEGER)
              for g in G for d in Days}

        YH = {(t, d): Heuristic.addVar(vtype=GRB.INTEGER, lb=MinWorkers,
                                       ub=MaxWorkers)
              for t in T for d in Days}

        # Dist[t, d] is the absolute distance between Curve[t, d] and Y[t, d]
        Dist = {(t, d): Heuristic.addVar() for t in T for d in Days}

        # Constraints
        for t in T:
            for d in Days:
                # Scheduling constraint
                Heuristic.addConstr(YH[t, d] == gp.quicksum(A[t][g] * XH[g, d]
                                                            for g in G))
                # Absolute value
                Heuristic.addConstr(Dist[t, d] >= Curve[d][t] - YH[t, d])
                Heuristic.addConstr(Dist[t, d] >= YH[t, d] - Curve[d][t])

        # Maxium working hours
        Heuristic.addConstr(gp.quicksum(L * YH[t, d] for t in T
                                        for d in Days) <= 60 * Data.MaxHours)
        # Solve Heuristic model
        Heuristic.setObjective(gp.quicksum(Dist[t, d] for t in T
                                           for d in Days) / len(T) * len(Days))
        Heuristic.setParam("OutputFlag", 0)
        Heuristic.optimize()

        # Retrieve the solution and give it to Gurobi
        HeuristicValue = 0
        for d in Days:
            for g in G:
                X[g, d].Start = round(XH[g, d].x)
            for t in T:
                Y[t, d].Start = round(YH[t, d].x)
                for xi in N:
                    if round(YH[t, d].x) == xi:
                        Z[xi, t, d].Start = 1
                    else:
                        Z[xi, t, d].Start = 0
            # Simulate the heuristic solution
            for s in S:
                ThetaH = Simulate(s, d, [(t, round(YH[t, d].x)) for t in T])
                for t in T:
                    HeuristicValue += ThetaH[t]
                    Theta[s, t, d].Start = ThetaH[t]

        HeuristicValue = HeuristicValue / len(S)
        HeuristicTime += time.time() - HeuristicStartTime

    # ----------------
    # --- Callback ---
    # ----------------
    # Best known objective value
    m._BestObj = GRB.INFINITY

    # Benders cuts added, and their sizes
    m._BendersCuts = 0
    CutSizes = []

    # Put the variables into lists
    YVar = list(Y.values())
    ZVar = list(Z.values())
    ThetaVar = list(Theta.values())

    def Callback(model, where):
        CallbackStartTime = time.time()
        if where == GRB.Callback.MIPSOL:

            # Retrieve the current solution
            YVal = {k: x for (k, x) in zip(Y.keys(),
                                           model.cbGetSolution(YVar))}
            ThetaVal = {k: x for (k, x) in zip(Theta.keys(),
                                               model.cbGetSolution(ThetaVar))}
            # Total delay of current solution
            SumDelay = 0

            # Simulate ech day of each scenario
            for s in S:
                for d in Days:

                    # Get the care worker levels of today
                    Levels = [round(YVal[t, d]) for t in T]

                    Delay = Simulate(s, d, [(t, Levels[t]) for t in T])
                    SumDelay += sum(Delay[t] for t in T)

                    for t in T:
                        # No cut required if Theta is high enough
                        if ThetaVal[s, t, d] >= Delay[t] - EPS:
                            continue

                        # Update Theta and log the cut
                        ThetaVal[s, t, d] = Delay[t]
                        m._BendersCuts += 1

                        # Get the starting neighborhood
                        tLow = t
                        tHigh = t

                        # Expand the neighborhood while legal
                        while True:
                            tLow = max(tLow - 1, 0)
                            tHigh = min(tHigh + 1, T[-1])
                            # If the neighborhood is good enough then stop
                            if Simulate(s, d, [(tt, Levels[tt]) for tt in
                                               range(tLow, tHigh + 1)])[t] >= \
                                    Delay[t] - EPS:
                                break

                        # Contract from the right if we can
                        while tHigh > t and \
                            Simulate(s, d, [(tt, Levels[tt]) for tt in
                                            range(tLow, tHigh - 1)])[t] >= \
                                Delay[t] - EPS:
                            tHigh -= 1

                        # Contract from the left if we can
                        while tLow < t and \
                            Simulate(s, d, [(tt, Levels[tt]) for tt in
                                            range(tLow + 1, tHigh)])[t] >= \
                                Delay[t] - EPS:
                            tLow += 1

                        # Base is a valid bound if we don't increase Y[t, d]
                        Base = Simulate(s, d, [(t, Levels[t])])[t]

                        # If we increase workers at t
                        Term1 = gp.quicksum((Delay[t] -
                                             Simulate(s, d, [(t, xi)])[t]) *
                                            Z[xi, t, d] for xi in
                                            N[N.index(Levels[t]) + 1:])

                        # If we dont increase workers at t
                        Term2 = (Delay[t] - Base) * (gp.quicksum(
                            Z[xi, tt, d] for tt in range(tLow, tHigh + 1)
                            for xi in N[N.index(Levels[tt]) + 1:] if tt != t))

                        # Add the Benders cut as a lazy constraint
                        model.cbLazy(Theta[s, t, d] >= Delay[t]-Term1-Term2)

                        # Log the size of the cut
                        CutSizes.append(int(tHigh - tLow + 1))

            # Primal heuristic
            if SumDelay < m._BestObj:

                # Store the better solution
                m._BestObj = SumDelay
                m._BestY = model.cbGetSolution(YVar)
                m._BestZ = model.cbGetSolution(ZVar)
                m._BestTheta = [ThetaVal[k] for k in Theta]

        # Pass the better solution to Gurobi
        if where == GRB.callback.MIPNODE and \
            model.cbGet(GRB.callback.MIPNODE_STATUS) == GRB.OPTIMAL and \
                model.cbGet(GRB.callback.MIPNODE_OBJBST) > m._BestObj + EPS:

            # Pass the better solution to Gurobi
            m.cbSetSolution(YVar, m._BestY)
            m.cbSetSolution(ZVar, m._BestZ)
            m.cbSetSolution(ThetaVar, m._BestTheta)

        m._TimeInCallback += time.time() - CallbackStartTime

    # ---------------
    # --- Results ---
    # ---------------
    # m.Params.Threads = 1 # (for testing)
    m.Params.LazyConstraints = 1
    m.Params.MIPGap = 0
    # TimeUpTo = time.time() - TotalTimeStart     #
    # m.Params.TimeLimit = round(3600 - TimeUpTo) # (for testing)
    m.optimize(Callback)
    TotalTime = time.time() - TotalTimeStart

    # Console output
    print(" ")
    print("--- Results ---")
    print(f"Solved instance to within {m.MIPGap*100} percent", end=" ")
    print(f"optimality in {round(TotalTime, 2)} seconds")
    print("Expected delay:", round(m.objVal, 3), "minutes")
    print("Optimal schedules")
    for d in Days:
        print(f"Day {d}:")
        for g in G:
            if X[g, d].x > 0.5:
                print(str(Shifts[g]) + ":", round(X[g, d].x))
    print(" ")

    # Save the results to a csv file
    with open(ResultsFile, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Scenarios used", len(S)])
        writer.writerow(["Model objective", m.objVal])
        writer.writerow(["MIP Gap", m.MIPGap])
        writer.writerow(["Solver time", m.Runtime])
        writer.writerow(["Initial time", InitialCutsTime])
        writer.writerow(["Heuristic time", HeuristicTime])
        writer.writerow(["Total time", TotalTime])
        writer.writerow(["Benders cuts", m._BendersCuts])
        writer.writerow(["Initial cuts", InitialCutsAdded])
        writer.writerow(["Heuristic val.", HeuristicValue])
        writer.writerow(["Simulations", len(Simulations)])
        writer.writerow(["Simulation time", m._SimulationTime])
        writer.writerow(["Time proportion", m._SimulationTime / TotalTime])
        writer.writerow(["Callback time", m._TimeInCallback])
        writer.writerow(["Size mean", np.mean(CutSizes)])
        writer.writerow(["Size std. dev.", np.std(CutSizes)])
        writer.writerow(["Node count", m.NodeCount])

    # Save the optimal solution to a csv file
    with open(SolutionFile, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time period"]+[f"Day {d}" for d in Days])
        for t in T:
            writer.writerow([str(t)]+[round(Y[t, d].x) for d in Days])

    # Save the cut sizes to a csv file
    with open(SizesFile, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Size"]+["Proportion" for d in Days])
        XAxis = list(range(1, len(T) + 1))
        Cuts = [0 for t in XAxis]
        TotalCuts = len(CutSizes)
        for c in CutSizes:
            Cuts[int(round(c - 1))] += 1
            YAxis = [c/TotalCuts for c in Cuts]
        for t in XAxis:
            writer.writerow([t, YAxis[int(t-1)]])
