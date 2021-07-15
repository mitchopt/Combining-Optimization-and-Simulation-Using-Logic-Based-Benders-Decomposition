# -*- coding: utf-8 -*-
import csv
import heapq
import networkx as nx
import time
import itertools as it
import numpy as np

# Helper functions
from alp_helpers import LoadNetwork, CalcFeasible

# Gurobi
import gurobipy as gp
from gurobipy import GRB


def Optimize(Data, NetworkFile, Scenario, Inits, ResultsFile, BruteForce):

    # Small float
    EPS = 0.00001

    # Load network
    G, P, Station, Hospital = LoadNetwork(NetworkFile)

    # Response time target
    Delta = Data.Delta

    # Max Ambulances per station
    M = Data.M

    # Total ambulances
    M1 = Data.M1

    # Set of ambulance levels
    N = range(M + 1)

    # Set of scenarios
    S = range(len(Scenario))

    # Set of nodes
    V = range(len(P))

    # Set of stations
    K = range(len(Station))

    # Set of hospitals
    E = range(len(Hospital))

    # Path[u][v] is the distance from node u to v
    Path = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))

    # Preference[v] is a list of stations ordered by proximity to v
    Preference = {v: sorted(K, key=lambda x: Path[Station[x]][v]) for v in V}

    # Closest[k] is the set of nodes whose first preference station is k
    Closest = {k: [v for v in V if Preference[v][0] == k] for k in K}

    # Total travel time from station k to node n
    StatToNode = {(k, v): Path[Station[k]][v] for k in K for v in V}

    # NodeToHosp[n, e] is the travel time from node n to hospital e
    NodeToHosp = {(v, e): Path[v][Station[e]] for v in V for e in E}

    # HospToStat[e, k] is the travel time from hospital e to station k
    HospToStat = {(e, k): Path[Station[e]][Station[k]] for e in E for k in K}

    # --------------------
    # --- Gurobi model ---
    # --------------------
    m = gp.Model()

    # Y[k] is the number of ambulances at station k
    Y = {k: m.addVar(ub=M, vtype=GRB.INTEGER) for k in K}

    # Z[xi, k] = 1 if and only if Y[k] = xi
    Z = {(xi, k): m.addVar(vtype=GRB.BINARY) for xi in N for k in K}

    # Benders variable on (s, k)
    Theta = {(s, k): m.addVar() for s in S for k in K}

    # Make sure the Z variables agree with the Y variables
    for k in K:
        m.addConstr(gp.quicksum(Z[n, k] for n in N) == 1)
        m.addConstr(gp.quicksum(Z[n, k]*n for n in N) == Y[k])

    # Total number of ambulances
    m.addConstr(gp.quicksum(Y[k] for k in K) == M1)

    # Objective function
    m.setObjective(gp.quicksum(Theta[s, k]/len(S) for s in S for k in K))

    # -------------------
    # --- Timekeeping ---
    # -------------------
    TotalTimeStart = time.time()
    InitialCutsTime = 0
    m._SimulationTime = 0
    m._TimeInCallback = 0

    # ---------------------------------
    # --- Discrete-Event Simulation ---
    # ---------------------------------
    _Simulations = {}

    def Simulate(s, LevelList):
        SimStartTime = time.time()

        # Retrieve levels
        Levels = [M for k in K]
        for k, l in LevelList:
            Levels[k] = l

        # Check the cache
        TupleLevels = tuple(Levels)
        if (s, TupleLevels) in _Simulations:
            return _Simulations[(s, TupleLevels)]

        # Peformance measures
        Missed = {k: 0 for k in K}

        # A[k] is the set of ambulances available at station k
        A = {k: [] for k in K}
        for k in K:
            for i in range(Levels[k]):
                heapq.heappush(A[k], 0)

        # K1 is the set of stations with at least one mabulance
        K1 = [k for k in K if Levels[k] > 0.9]

        # Dispatch an ambulance to call c
        for (i, c) in enumerate(Scenario[s]):

            # kk reaches the call first and r is the time it does
            r, kk = min((max(c[0], A[k][0]) + c[2] +
                         StatToNode[k, c[1]], k) for k in K1)

            # Remove it from the queue
            heapq.heappop(A[kk])

            # Calculate return time
            if c[4]:
                re, ee = min((NodeToHosp[c[1], e], e) for e in E)
                return_time = r + c[3] + re + c[5] + HospToStat[ee, k]
            else:
                return_time = r + c[3] + StatToNode[kk, c[1]]

            # Push the ambulance back onto the queue
            heapq.heappush(A[kk], return_time)

            # Update peformance
            Missed[Preference[c[1]][0]] += int(r - c[0] > Delta)

        # Cache and return the peformance
        _Simulations[s, TupleLevels] = Missed
        m._SimulationTime += time.time() - SimStartTime
        return Missed

    # ------------------------
    # ----- Initial cuts -----
    # ------------------------
    InitialCutsAdded = 0
    if Inits:
        print("Generating initial cuts")
        print("Scenario:", end=" ")
        InitialCutsStartTime = time.time()

        # Pairs[k] contains the stations we will cut on k with
        Pairs = {k: [] for k in K}
        for k in K:
            for v in Closest[k]:
                if Preference[v][1] not in Pairs[k]:
                    Pairs[k].append(Preference[v][1])

        for s in S:
            print(s, end=" ")
            # D[k, xi_1] is the peformance of
            # k when k has xi_1 ambulances
            D = {}

            # D2[k, kk, xi_1, xi_2] is the peformance of
            # k when k has xi_1 ambulances and kk has xi_2
            D2 = {}

            # Populate D1 and D2
            for k in K:
                for xi_1 in N:
                    D[k, xi_1] = Simulate(s, [(k, xi_1)])[k]
                    for l in Pairs[k]:
                        for xi_2 in N:

                            # Initial guess for D2
                            D2[k, l, xi_1, xi_2] = D[k, xi_1]

                        for xi_2 in N[:-1]:
                            D2[k, l, xi_1, xi_2] = \
                                Simulate(s, [(k, xi_1), (l, xi_2)])[k]

                            # If we reach D1[k, xi_1] we can stop
                            LL = [(k, xi_1), (l, xi_2)]
                            if Simulate(s, LL)[k] <= D[k, xi_1] + EPS:
                                break

                # Add the cuts
                for l in Pairs[k]:
                    for xi_1 in N:
                        # First initial cut
                        m.addConstr(Theta[s, k] >=
                                    gp.quicksum(D2[k, l, xi_1, xi_2]*Z[xi_2, l]
                                                for xi_2 in N) -
                                    gp.quicksum(max(D2[k, l, xi_1, xi_2] -
                                                    D2[k, l, xi_p, xi_2]
                                                    for xi_2 in N)*Z[xi_p, k]
                                                for xi_p in N[xi_1 + 1:]))
                        InitialCutsAdded += 1

                    for xi_2 in N:
                        # Second initial cut
                        m.addConstr(Theta[s, k] >=
                                    gp.quicksum(D2[k, l, xi_1, xi_2]*Z[xi_1, k]
                                                for xi_1 in N) -
                                    gp.quicksum(max(D2[k, l, xi_1, xi_2] -
                                                    D2[k, l, xi_1, xi_p]
                                                    for xi_1 in N)*Z[xi_p, l]
                                                for xi_p in N[xi_2 + 1:]))
                        InitialCutsAdded += 1
        InitialCutsTime = time.time() - InitialCutsStartTime

    # --------------------
    # ----- Callback -----
    # --------------------
    # Benders cuts added
    m._BendersCuts = 0

    # Best known objective
    m._BestObj = GRB.INFINITY

    # Sizes of cuts added
    CutSizes = []

    # Listify the variables
    YVar = list(Y.values())
    ZVar = list(Z.values())
    TVar = list(Theta.values())

    def Callback(model, where):
        CallbackStartTime = time.time()
        if where == GRB.Callback.MIPSOL:

            # Retrieve the current solution
            YVal = {k: x for (k, x) in zip(Y.keys(),
                                           model.cbGetSolution(YVar))}
            TVal = {k: x for (k, x) in zip(Theta.keys(),
                                           model.cbGetSolution(TVar))}
            # Total peformance of solution
            SumObj = 0

            # Simulate each scenario with current solution
            Levels = [round(YVal[k]) for k in K]
            for s in S:

                # Simulate the current solution
                Missed = Simulate(s, [(k, Levels[k]) for k in K])
                SumObj += sum(Missed[k] for k in K)

                for k in K:

                    # If Theta[s, k] is correct then no cut needed
                    if TVal[s, k] >= Missed[k] - EPS:
                        continue

                    # Update ThetaVal
                    TVal[s, k] = Missed[k]
                    m._BendersCuts += 1

                    # Neighborhood generation
                    Neigh = [k]
                    for j in range(len(K) - 1):

                        # Add the (j + 1)st preference stations
                        for v in Closest[k]:
                            if Preference[v][j + 1] not in Neigh:
                                Neigh.append(Preference[v][j + 1])

                        # Test the new window
                        NeighLevels = [(kk, Levels[kk]) for kk in Neigh]
                        if Simulate(s, NeighLevels)[k] >= Missed[k] - EPS:
                            break

                    # Contract Neigh if we can
                    NeighIter = Neigh.copy()
                    NeighIter.reverse()
                    for j in NeighIter:
                        if j != k:

                            # Copy of neighborhood
                            NeighCopy = Neigh.copy()
                            NeighCopy.remove(j)
                            NeighCLL = [(kk, Levels[kk]) for kk in NeighCopy]
                            if Simulate(s, NeighCLL)[k] >= Missed[k] - EPS:
                                Neigh.remove(j)

                    # Base is a lower bound if we don't increase Y[k]
                    Base = Simulate(s, [(k, Levels[k])])[k]

                    # Increase ambulances at kth station
                    Term1 = gp.quicksum((Missed[k] -
                                         Simulate(s, [(k, xi)])[k])*Z[xi, k]
                                        for xi in N[N.index(Levels[k]) + 1:])

                    # Don't increase ambulances kth station
                    Term2 = (Missed[k]-Base) * \
                        gp.quicksum(Z[xi, kk] for kk in Neigh if kk != k
                                    for xi in N[N.index(Levels[kk]) + 1:])

                    # Add the cut as a lazy constraint
                    model.cbLazy(Theta[s, k] >= Missed[k] - Term1 - Term2)
                    CutSizes.append(len(Neigh))

            # Primal heuristic
            CurrentObj = SumObj / len(S)
            if CurrentObj < m._BestObj:

                # Store the better solution
                m._BestObj = CurrentObj
                m._BestY = model.cbGetSolution(YVar)
                m._BestZ = model.cbGetSolution(ZVar)
                m._BestTheta = [TVal[key] for key in Theta]

        # We can only pass new solutions to gurobi inside MIPNODE
        if where == GRB.callback.MIPNODE and \
            model.cbGet(GRB.callback.MIPNODE_STATUS) == GRB.OPTIMAL and \
                model.cbGet(GRB.callback.MIPNODE_OBJBST) > m._BestObj + EPS:
            m.cbSetSolution(YVar, m._BestY)
            m.cbSetSolution(ZVar, m._BestZ)
            m.cbSetSolution(TVar, m._BestTheta)

        m._TimeInCallback += time.time() - CallbackStartTime

    # ---------------
    # --- Results ---
    # ---------------
    m.Params.Threads = 1
    m.Params.LazyConstraints = 1
    m.Params.MipGap = 0
    TimeUpTo = time.time() - TotalTimeStart
    TimeLim = 3600 - TimeUpTo
    m.Params.TimeLimit = TimeLim
    m.optimize(Callback)
    TotalTime = time.time() - TotalTimeStart

    # Optimal solution
    YOpt = [(k, round(Y[k].x)) for k in K]

    # ----------------------
    # --- Console output ---
    # ----------------------
    print(" ")
    print("--- Results ---")
    print(f"Solved instance to within {m.MIPGap*100} percent", end=" ")
    print(f"optimality in {round(TotalTime, 2)} seconds")
    print("Average calls:", sum(len(Scenario[s]) for s in S) / len(S))
    print("Model objective (calls missed):", round(m.objVal, 3))

    # Simulation value
    TrueObj = sum(Simulate(s, YOpt)[k] for s in S for k in K) / len(S)
    print("True value (from simulation):", round(TrueObj, 3))

    # Model error
    MError = abs(m.objVal - TrueObj) / TrueObj
    print("Model error:", MError)

    # Calculate the number of fesible solutions
    FeasibleSolutions = CalcFeasible(len(K), M, M1)

    # Proportion simulated
    SimProportion = float(len(_Simulations)/FeasibleSolutions)

    # Brute force the solution
    # Only valid if M = 1
    if BruteForce:
        BruteForceStart = time.time()
        Solutions = list(it.combinations(list(K), M1))
        Values = {sol: 0 for sol in Solutions}
        i = 0
        total = len(Solutions)
        for sol in Solutions:
            print("To go:", total - i)
            i += 1
            TestLL = [(k, 0) for k in K if k not in sol]
            for k in K:
                if k in sol:
                    TestLL.append((k, 1))
            for s in S:
                Values[sol] += sum(Simulate(s, TestLL)[k] for k in K)
            Values[sol] = Values[sol]/len(S)
        BruteForceSolution = min(Values[sol] for sol in Values)
        BruteForceTime = time.time() - BruteForceStart

    # Save results to a csv file
    with open(ResultsFile, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Scenarios", len(S)])
        writer.writerow(["Model objective", m.objVal])
        writer.writerow(["MIP Gap", m.MIPGap])
        writer.writerow(["True objective", TrueObj])
        writer.writerow(["Model error", MError])
        writer.writerow(["Initial time", InitialCutsTime])
        writer.writerow(["Solver time", m.Runtime])
        writer.writerow(["Simulation time", m._SimulationTime])
        writer.writerow(["Time in callback", m._TimeInCallback])
        writer.writerow(["Total time", TotalTime])
        writer.writerow(["Benders cuts", m._BendersCuts])
        writer.writerow(["Initial cuts", InitialCutsAdded])
        writer.writerow(["Feasible solutions", FeasibleSolutions])
        writer.writerow(["Sim proportion", SimProportion])
        writer.writerow(["Time proportion", m._SimulationTime / TotalTime])
        writer.writerow(["Cut size mean", np.mean(CutSizes)])
        writer.writerow(["Cut size var", np.std(CutSizes)])
        writer.writerow(["Node count", m.NodeCount])
        if BruteForce:
            writer.writerow(["Brute force solution", BruteForceSolution])
            BError = abs(TrueObj - BruteForceSolution) / BruteForceSolution
            writer.writerow(["Brute force error", BError])
            writer.writerow(["Brute force time", BruteForceTime])
