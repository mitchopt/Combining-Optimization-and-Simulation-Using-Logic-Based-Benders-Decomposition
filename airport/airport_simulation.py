# -*- coding: utf-8 -*-
import time
import heapq


# A simulation engine with caching 
# and other efficiencies
class Airport_Simulation:
    def __init__(self, T, L, MaxDesks, Scenarios):
        
        self.Simulations = {}
        self.PassStart = {}
        self.DeskStart = {}
        self.T = T
        self.L = L
        self.MaxDesks = MaxDesks
        self.Scenarios = Scenarios
        
        # Initialize data structures
        for s in range(len(Scenarios)):
            self.Simulate(s, None)
    
    
    # Simulate scenario s with counter levels (t, Y[t])
    def Simulate(self, s, LevelsList):
        # Note the start time of the simulation
        sim_start = time.time()
    
        # Set up counter levels
        Levels = [self.MaxDesks for t in self.T]
        init = False
        # Trigger initialization
        if not LevelsList:
            init = True
        else:
            for (t, level) in LevelsList:
                Levels[t] = level
    
        # Check the cache
        TupleLevels = tuple(Levels)
        if (s, TupleLevels) in self.Simulations:
            return self.Simulations[s, TupleLevels]
    
        # Performance measures
        QueuingTime = {t: 0 for t in self.T}
    
        # Stopping criteria
        if init:
            tEnd = self.T[-1] + 1
        else:
            tEnd = LevelsList[-1][0] + 1
    
        # Set up the simulation
        if init or LevelsList[0][0] == 0:
            tUpto = 0
            Desks = []
            for i in range(Levels[0]):
                heapq.heappush(Desks, 0)
            Arrivals = self.Scenarios[s]
    
        # Get the starting state
        else:
            tUpto = LevelsList[0][0]
            Desks = list(self.DeskStart[s, tUpto])
            for i in range(Levels[tUpto], self.MaxDesks):
                heapq.heappop(Desks)
            Arrivals = self.Scenarios[s][self.PassStart[s, tUpto]:]
    
        # Schedule the ith passenger
        for (i, k) in enumerate(Arrivals):
    
            # Stopping criteria
            if k[0] > tEnd * self.L:
                break
            # Boundary condition
            OverFlow = False
    
            # If initializing then populate the data structures
            if init:
                # If the next passenger doesn't arrive until the next
                if k[0] >= self.L * (tUpto + 1):  # time period then move on
                    tUpto += 1
                    self.DeskStart[s, tUpto] = tuple(Desks)
                    self.PassStart[s, tUpto] = i
    
            # Open and shut counters if necessary
            else:
                # If the next passenger hasn't arrived yet then move on
                while len(Desks) == 0 or max(k[0], Desks[0]) >= self.L*(tUpto + 1):
                    # If we run out of time without finishing
                    if tUpto >= self.T[-1]:
                        OverFlow = True
                        break
                    # Increase the counter levels if necessary
                    if Levels[tUpto + 1] > Levels[tUpto]:
                        for i in range(Levels[tUpto], Levels[tUpto + 1]):
                            heapq.heappush(Desks, self.L*(tUpto + 1))
                    # Decrease the counter levels if necessary
                    if Levels[tUpto + 1] < Levels[tUpto]:
                        for i in range(Levels[tUpto + 1], Levels[tUpto]):
                            heapq.heappop(Desks)
                    # Move on to the next time period
                    tUpto += 1
    
            # If OverFlow was triggered then incur a penalty
            if OverFlow:
                QueuingTime[k[0] // self.L] += len(self.T) * self.L
    
            else:
                # Get the next counter from the queue
                Desk = heapq.heappop(Desks)
                # Compute the queuing time
                ThisQTime = max(0, Desk - k[0])
                # Update the performance
                QueuingTime[k[0] // self.L] += ThisQTime
                # Get the next idle time of the counter
                Desk = max(Desk, k[0]) + k[2]
                # Put the desk back on the queue
                heapq.heappush(Desks, Desk)
    
        # When initializing, if we finish scheduling the passengers before
        if init:  # we get to the end, then finish populating the
            for tUpto in self.T[tUpto + 1:]:  # data structures
                self.DeskStart[s, tUpto] = tuple(Desks)
                self.PassStart[s, tUpto] = len(self.Scenarios[s])
    
        # Cache and return the performance
        self.Simulations[s, TupleLevels] = QueuingTime
        return QueuingTime
     

