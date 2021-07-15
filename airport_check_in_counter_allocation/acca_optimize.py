# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB
import heapq
import time
import csv


# Optimize the ACCA problem
def Optimize(Data, inits, Scenario, Method, Inits, ResultsFile, SolutionFile):

    # Small number
    EPS = 0.0001

    # Number of time periods
    H = Data.H
    # Length of a time period
    L = Data.L
    # Set of time periods
    T = range(H)
    # Max check-in counters
    MaxDesks = Data.MaxDesks
    # Set of counter levels
    N = range(MaxDesks + 1)
    # Desk cost
    DeskCost = Data.DeskCost
    # Queueing cost per hour
    QueueCost = Data.QueueCost
    # Set of scenarios
    S = Data.S

    # --------------------
    # --- Gurobi model ---
    # --------------------
    m = gp.Model()

    # Y[t] is the number of check-in counters open at time t
    Y = {t: m.addVar(ub=MaxDesks, vtype=GRB.INTEGER) for t in T}

    # Z[xi, t]=1 if and only if Y[t] = xi
    Z = {(xi, t): m.addVar(vtype=GRB.BINARY) for xi in N for t in T}

    # Theta[s, t] estiamtes the waiting time of passengers
    Theta = {(s, t): m.addVar() for s in S for t in T}

    # Constraints
    for t in T:

        # Link the Y and Z variables
        m.addConstr(gp.quicksum(Z[xi, t] for xi in N) == 1)
        m.addConstr(gp.quicksum(Z[xi, t] * xi for xi in N) == Y[t])

        # Total processing time of some passengers about t
        After = max(sum(k[2] for k in Scenario[s] if k[0] >= t*L) for s in S)
        Before = max(sum(k[2] for k in Scenario[s]
                         if Data.Start[k[1]] + len(Data.Arrive) <= t)
                     for s in S)

        # Open enough counters to clear the passengers
        m.addConstr(L * gp.quicksum(Y[tt] for tt in T[t:]) >= After)
        m.addConstr(L * gp.quicksum(Y[tt] for tt in T[:t + 1]) >= Before)

    # Objective function
    m.setObjective(DeskCost * gp.quicksum(Y[t] for t in T) +
                   QueueCost * gp.quicksum(Theta[s, t]/(L * len(S))
                                           for s in S for t in T))

    # ---------------------------------
    # --- Discrete-Event Simulation ---
    # ---------------------------------
    PassStart = {}
    DeskStart = {}
    Simulations = {}
    m._sim_time = 0

    # Simulate scenario s with counter levels (t, Y[t])
    def Simulate(s, LevelsList):
        # Note the start time of the simulation
        sim_start = time.time()

        # Set up counter levels
        Levels = [MaxDesks for t in T]
        init = False
        # Trigger initialization
        if not LevelsList:
            init = True
        else:
            for (t, level) in LevelsList:
                Levels[t] = level

        # Check the cache
        TupleLevels = tuple(Levels)
        if (s, TupleLevels) in Simulations:
            return Simulations[s, TupleLevels]

        # Performance measures
        QueuingTime = {t: 0 for t in T}

        # Stopping criteria
        if init:
            tEnd = T[-1] + 1
        else:
            tEnd = LevelsList[-1][0] + 1

        # Set up the simulation
        if init or LevelsList[0][0] == 0:
            tUpto = 0
            Desks = []
            for i in range(Levels[0]):
                heapq.heappush(Desks, 0)
            Arrivals = Scenario[s]

        # Get the starting state
        else:
            tUpto = LevelsList[0][0]
            Desks = list(DeskStart[s, tUpto])
            for i in range(Levels[tUpto], MaxDesks):
                heapq.heappop(Desks)
            Arrivals = Scenario[s][PassStart[s, tUpto]:]

        # Schedule the ith passenger
        for (i, k) in enumerate(Arrivals):

            # Stopping criteria
            if k[0] > tEnd * L:
                break
            # Boundary condition
            OverFlow = False

            # If initializing then populate the data structures
            if init:
                # If the next passenger doesn't arrive until the next
                if k[0] >= L * (tUpto + 1):  # time period then move on
                    tUpto += 1
                    DeskStart[s, tUpto] = tuple(Desks)
                    PassStart[s, tUpto] = i

            # Open and shut counters if necessary
            else:
                # If the next passenger hasn't arrived yet then move on
                while len(Desks) == 0 or max(k[0], Desks[0]) >= L*(tUpto + 1):
                    # If we run out of time without finishing
                    if tUpto >= T[-1]:
                        OverFlow = True
                        break
                    # Increase the counter levels if necessary
                    if Levels[tUpto + 1] > Levels[tUpto]:
                        for i in range(Levels[tUpto], Levels[tUpto + 1]):
                            heapq.heappush(Desks, L*(tUpto + 1))
                    # Decrease the counter levels if necessary
                    if Levels[tUpto + 1] < Levels[tUpto]:
                        for i in range(Levels[tUpto + 1], Levels[tUpto]):
                            heapq.heappop(Desks)
                    # Move on to the next time period
                    tUpto += 1

            # If OverFlow was triggered then incur a penalty
            if OverFlow:
                QueuingTime[k[0] // L] += len(T) * L

            else:
                # Get the next counter from the queue
                Desk = heapq.heappop(Desks)
                # Compute the queuing time
                ThisQTime = max(0, Desk - k[0])
                # Update the performance
                QueuingTime[k[0] // L] += ThisQTime
                # Get the next idle time of the counter
                Desk = max(Desk, k[0]) + k[2]
                # Put the desk back on the queue
                heapq.heappush(Desks, Desk)

        # When initializing, if we finish scheduling the passengers before
        if init:  # we get to the end, then finish populating the
            for tUpto in T[tUpto + 1:]:  # data structures
                DeskStart[s, tUpto] = tuple(Desks)
                PassStart[s, tUpto] = len(Scenario[s])

        # Cache and return the performance
        Simulations[s, TupleLevels] = QueuingTime
        m._sim_time += time.time() - sim_start
        return QueuingTime

    # Initialize
    for s in S:
        Simulate(s, None)

    # --------------------
    # --- Initial Cuts ---
    # --------------------
    initial_cuts = 0
    initial_time = 0
    if Inits:
        initial_start = time.time()
        print("# ---Generating initial cuts---")
        print("# Scenario:", end=" ")
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
                    QueuingTime = Simulate(s, [(t, xi_1)])
                    W1[t, xi_1] = QueuingTime[t]

                    # Initial values for W2
                    for xi_2 in N:
                        W2[t, xi_2, xi_1] = QueuingTime[t]

                    if t > 0 and xi_1 < N[-1]:
                        for xi_2 in N[:-1]:

                            # Fix the 2D queuing times
                            QueuingTime = Simulate(s, [(t-1, xi_2), (t, xi_1)])
                            W2[t, xi_2, xi_1] = QueuingTime[t]
                            if QueuingTime[t] <= W1[t, xi_1] + EPS:
                                break

                    # We can stop fixing W2 if there is no delay
                    elif QueuingTime[t] == 0:
                        break

                # Add cuts
                for xi_1 in N:
                    if t > 0:
                        initial_cuts += 2

                        # First initial cut
                        m.addConstr(Theta[s, t] >=
                                    gp.quicksum(Z[xi_2, t-1]*W2[t, xi_2, xi_1]
                                                for xi_2 in N) -
                                    gp.quicksum(max(W2[t, xi_2, xi_1] -
                                                    W2[t, xi_2, xi_p]
                                                    for xi_2 in N)*Z[xi_p, t]
                                                for xi_p in N[xi_1+1:]))
                for xi_2 in N:
                    if t > 0:

                        # Second initial cut
                        m.addConstr(Theta[s, t] >=
                                    gp.quicksum(Z[xi_1, t]*W2[t, xi_2, xi_1]
                                                for xi_1 in N) -
                                    gp.quicksum(max(W2[t, xi_2, xi_1] -
                                                    W2[t, xi_p, xi_1]
                                                    for xi_1 in N)*Z[xi_p, t-1]
                                                for xi_p in N[xi_2+1:]))
        # Time spent generating initial cuts
        initial_time += time.time() - initial_start
        print("Done \n")

    # ----------------
    # --- Callback ---
    # ----------------
    m._BestObj = GRB.INFINITY
    m._lazy_constraints = 0

    # Put the variables into lists
    YVar = list(Y.values())
    ZVar = list(Z.values())
    ThetaVar = list(Theta.values())

    def Callback(model, where):

        # Integer solutions
        if where == GRB.Callback.MIPSOL:

            # Retrieve the current solution
            YVal = {k: x for (k, x) in zip(Y.keys(),
                                           model.cbGetSolution(YVar))}
            ThetaVal = {k: x for (k, x) in zip(Theta.keys(),
                                               model.cbGetSolution(ThetaVar))}

            # Get the current levels
            Levels = [round(YVal[t]) for t in T]

            # Total performance of the current solution
            SumDelay = 0

            # Cuts on s
            for s in S:

                # Simulate the current solution; note total performance
                QueueTime = Simulate(s, [(t, Levels[t]) for t in T])
                SumDelay += sum(QueueTime[t] for t in T)

                # Cuts on t
                for t in T:

                    # If Theta[s, t] is correct then move on
                    if ThetaVal[s, t] >= QueueTime[t] - EPS:
                        continue

                    # Updata Thetas
                    ThetaVal[s, t] = QueueTime[t]

                    # Log the cut we're about to add
                    m._lazy_constraints += 1

                    # Add Benders cuts
                    # We only implemented all four cuts for the ACCA problem

                    if Method == 0:
                        # -------------------
                        # --- No good cut ---
                        # -------------------
                        model.cbLazy(Theta[s, t] >= QueueTime[t] *
                                     (1-gp.quicksum(Z[xi, tt]
                                                    for tt in T for xi in N
                                                    if xi != Levels[tt])))

                    elif Method == 1:
                        # ------------------------
                        # --- Monotonicity cut ---
                        # ------------------------
                        model.cbLazy(Theta[s, t] >= QueueTime[t] *
                                     (1-gp.quicksum(Z[xi, tt] for tt in T for
                                                    xi in N[Levels[tt] + 1:])))

                    elif Method == 2 or Method == 3:
                        # --------------------
                        # --- Neighborhood ---
                        # --------------------
                        tLow = t
                        tHigh = t

                        #
                        # If you're reading this: sorry about the long lines!
                        # Compressing them became messy and looked ugly
                        #

                        # Two-sided window expansion
                        while True:
                            tLow = max(tLow - 1, 0)
                            tHigh = min(tHigh + 1, T[-1])
                            if Simulate(s, [(tt, Levels[tt]) for tt in range(tLow, tHigh + 1)])[t] >= QueueTime[t] - EPS:
                                break

                        # Contract from the right if we can
                        while tHigh > t and Simulate(s, [(tt, Levels[tt]) for tt in range(tLow, tHigh - 1)])[t] >= QueueTime[t] - EPS:
                            tHigh -= 1

                        # Contract from the left if we can
                        while tLow < t and Simulate(s, [(tt, Levels[tt]) for tt in range(tLow + 1, tHigh)])[t] >= QueueTime[t] - EPS:
                            tLow += 1

                        if Method == 2:
                            # -----------------
                            # --- Local cut ---
                            # -----------------
                            model.cbLazy(Theta[s, t] >= QueueTime[t] * (1 - gp.quicksum(Z[xi, tt] for tt in range(tLow, tHigh + 1) for xi in N[N.index(Levels[tt]) + 1:])))

                        elif Method == 3:
                            # ------------------------
                            # --- Strengthened cut ---
                            # ------------------------
                            # Base is a valid  bound if we don't increase y[k]
                            Base = Simulate(s, [(t, Levels[t])])[t]

                            # Increase levels at t
                            Term1 = gp.quicksum((QueueTime[t] - Simulate(s, [(t, xi)])[t]) * Z[xi, t] for xi in N[N.index(Levels[t]) + 1:])

                            # Increase levels at others
                            Term2 = (QueueTime[t] - Base) * (gp.quicksum(Z[xi, tt] for tt in range(tLow, tHigh + 1) for xi in N[N.index(Levels[tt]) + 1:] if tt != t))

                            # Add the cut
                            model.cbLazy(Theta[s, t] >= QueueTime[t]
                                         - Term1 - Term2)

            # Primal heuristic
            CurrentObj = sum(Levels)*DeskCost + SumDelay*QueueCost/(L * len(S))
            if CurrentObj < m._BestObj:

                # Store the better solution
                m._BestObj = CurrentObj
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

    # ---------------
    # --- Results ---
    # ---------------
    # m.Params.Threads = 1 # (For testing)
    m.Params.LazyConstraints = 1
    m.Params.MIPGap = 0
    # m.Params.TimeLimit = round(3600 - initial_time) # (for testing)
    m.optimize(Callback)

    # Console output
    print("--- Results ---")
    print(f"Solved instance to within {m.MIPGap*100} percent", end=" ")
    print(f"optimality in {round(initial_time + m.Runtime, 2)} seconds")
    print("Operational cost:", DeskCost * round(sum(Y[t].x for t in T)))
    print("Average queuing time:",
          sum(Theta[s, t].x for s in S for t in T) / len(S), "minutes")
    print("Service cost:",
          QueueCost * sum(Theta[s, t].x for s in S for t in T) / (L * len(S)))
    print("Total cost:", round(m.objVal, 3), "minutes")

    # Model objective value
    ModelObj = m.objVal

    # Model gap
    MIPGap = m.MIPGap

    # Optimization time
    OptTime = m.Runtime

    # BendersCuts
    BendersCuts = m._lazy_constraints

    # Initial cuts added
    InitialCuts = initial_cuts

    # Initial cuts time
    InitialTime = initial_time

    # Proportion of time spent in simulation
    TimeProportion = m._sim_time / (initial_time + m.Runtime)

    # Save the results to a csv file
    with open(ResultsFile, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Scenarios", len(S)])
        writer.writerow(["Model objective", ModelObj])
        writer.writerow(["MIP Gap", MIPGap])
        writer.writerow(["Solver time", OptTime])
        writer.writerow(["Initial time", InitialTime])
        writer.writerow(["Total time", InitialTime + OptTime])
        writer.writerow(["Time proportion", TimeProportion])
        writer.writerow(["Benders cuts", BendersCuts])
        writer.writerow(["Initial cuts", InitialCuts])
        writer.writerow(["Simulations", len(Simulations)])

    # Save the optimal solution to a csv file
    with open(SolutionFile, 'w', newline='') as file:
        writer = csv.writer(file)
        for t in T:
            writer.writerow([str(t), round(Y[t].x)])
