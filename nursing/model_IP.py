# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB
from helper_functions import GenShifts, GenMatrix


def nursing_IP(Requests, H, Night, T, L, Lengths, MinWorkers, MaxWorkers, MaxHours, YFix):

    # (*) Extended set of time periods (the night shift is one long period)
    TT = range(len(T) + 1)

    # (*) Set of minutes during the day
    Times = range(H * L)

    # (*) Extended set of minutes (including night shift)
    TimesN = range((H + Night)*L)

    # BigM
    BigM = MaxWorkers - MinWorkers

    # (*) The set of shift types
    Shifts = GenShifts(Lengths, H, L)

    # (*) The matrix A linking shifts to levels
    A = GenMatrix(Shifts, L, T)

    # -------------------------------
    # --- Helpful data structures ---
    # -------------------------------
    
    # InPeriod[minute] is the period containing minute 'minute'
    # The night shift is one big time period
    InPeriod = {minute: TT[-1] for minute in TimesN}
    for minute in Times:
        InPeriod[minute] = minute // L
    
    # Set of shifts
    G = range(len(Shifts))
    
    # Set of jobs
    J = range(len(Requests))
    
    
    # CanStart[j] is the set of minutes that job j can start
    # Any time from release time, until last minute minus processing time
    CanStart = {j: range(
        Requests[j][0], TimesN[-Requests[j][1]]) for j in J}

    # Set of jobs released so far at each minute
    Avail = {minute: [j for j in J if Requests[j][0] <= minute] for minute in TimesN}

    
    # --------------------
    # --- Gurobi model ---
    # --------------------
    m = gp.Model()
    
    # The number of care workers working shift g on day d
    X = {g: m.addVar(vtype=GRB.INTEGER) for g in G}
    
    # The number of care workers working in time period t
    Y = {t: m.addVar(vtype=GRB.INTEGER, lb=MinWorkers, ub=MaxWorkers) for t in T}
    Y[TT[-1]] = m.addVar(vtype=GRB.INTEGER)
    m.addConstr(Y[TT[-1]] == MinWorkers)
    
    for t in T:  # Link shifts to agents
        m.addConstr(Y[t] == gp.quicksum(A[t][g] * X[g] for g in G))
    
    # Maximum working hours
    m.addConstr(gp.quicksum(L * Y[t] for t in T) <= 60 * MaxHours)
    
    # Fix solution if given
    if YFix is not None:
        for t in YFix:
            m.addConstr(Y[t] == YFix[t])

    # Binary indicator that a job  j
    # has started by minute minute
    Sigma = {(j, minute): m.addVar(
        vtype=GRB.BINARY) for j in J for minute in TimesN}
    
    # -1 index
    for j in J:
        Sigma[j, -1] = 0
    
    for j in J:  # No job starts done
        m.addConstr(Sigma[j, min(CanStart[j]) - 1] == 0)
    
    # Constraints per job
    for j in J:
        
        # Finish every job
        m.addConstr(Sigma[j, max(CanStart[j])] == 1)
        
        # Constraints per minute
        for minute in TimesN:
            
            # Correct start times
            if minute - 1 in TimesN:
                m.addConstr(Sigma[j, minute - 1] <= Sigma[j, minute])
                
            # FCFS Schedule
            if j - 1 in J:
                m.addConstr(Sigma[j, minute] <= Sigma[j - 1, minute])
                
            # Expression for total active jobs at minute
            ActiveJobs = gp.quicksum(
                Sigma[jj, minute] -
                Sigma[jj, max(minute - Requests[jj][1], Requests[jj][0] - 1)]
                for jj in Avail[minute] if jj != j)

            # Add the constraint
            if minute - 1 in TimesN:
                m.addConstr(Sigma[j, minute] - Sigma[j, minute - 1] <= 
                    Y[InPeriod[minute]] - ActiveJobs 
                    + BigM * (1 - Sigma[j, minute] + Sigma[j, minute - 1]))
            
            
    # Minimize total delay
    m.setObjective(gp.quicksum((minute - Requests[j][0])*(Sigma[j, minute] - Sigma[j, minute - 1])
                    for j in J for minute in CanStart[j]), GRB.MINIMIZE)
    
    
    
    # m.setParam("OutputFlag", 0)
    m.optimize()
    return m, {t: round(Y[t].x) for t in T}
    
    
    
    
    
    
