# -*- coding: utf-8 -*-
import time
import heapq


# A simulation engine with caching 
# and other efficiencies
class Nursing_Simulation:
    def __init__(self, T, L, MinWorkers, MaxWorkers, Scenario):
        
        self.Simulations = {}
        self.CareStart = {}
        self.ReqsStart = {}
        self.TimeInSimulation = 0
        self.T = T
        self.L = L
        self.MinWorkers = MinWorkers
        self.MaxWorkers = MaxWorkers
        self.Scenario = Scenario
        
        # Initialize data structures
        for s in range(len(Scenario)):
            self.Simulate(s, None)
    
    def Simulate(self, s, LevelsList):
        SimStartTime = time.time()
        

        # Set up the worker levels
        Levels = [self.MaxWorkers for t in self.T]
        init = False
        if not LevelsList:
            init = True
        else:
            for (t, l) in LevelsList:
                Levels[t] = l

        # Check the cache
        TupleLevels = tuple(Levels)
        if (s, TupleLevels) in self.Simulations:
            return self.Simulations[s, TupleLevels]

        # Total delays
        Delay = {t: 0 for t in self.T}

        # Stopping criteria
        if init:
            tEnd = self.T[-1] + 1
        else:
            tEnd = LevelsList[-1][0] + 1

        # Initialise the simulation
        if init or LevelsList[0][0] == 0:
            tUpto = 0
            Agents = []
    
            # Push each starting agent
            for i in range(Levels[0]):
                heapq.heappush(Agents, 0)
                
            # Get requests in this scenario
            Requests = self.Scenario[s]

        # If not initialising then
        # retrieve the starting state
        else:
            tUpto = LevelsList[0][0]
            Agents = []
            for a in list(self.CareStart[s, tUpto]):
                heapq.heappush(Agents, a)
            for i in range(int(Levels[tUpto]), self.MaxWorkers):
                heapq.heappop(Agents)
            Requests = self.Scenario[s][self.ReqsStart[s, tUpto]:]

        # For when we get to the nightshift
        NightShift = False

        # Service the ith request
        for (i, k) in enumerate(Requests):
            # print(i, end = " ")
            # Stopping condition
            if k[0] > tEnd * self.L:
                break

            # Initialization
            if init:
                # If the next request doesn't arrive until the next
                if k[0] >= (tUpto + 1) * self.L:  # time period, then move on
                    tUpto += 1

                    # Populate the data structures
                    self.CareStart[s, tUpto] = tuple(Agents)
                    self.ReqsStart[s, tUpto] = i

            else:
                # If the next time period isn't the night shift
                if round(tUpto + 1) in self.T:

                    # If there are no workers available, or the next
                    #  request hasen't arrived yet, then move on
                    while round(tUpto + 1) in self.T and \
                            max(k[0], Agents[0]) >= (tUpto + 1) * self.L:

                        # Add care workers
                        if Levels[tUpto + 1] > Levels[tUpto]:
                            for _ in range(Levels[tUpto], Levels[tUpto + 1]):
                                heapq.heappush(Agents, self.L*(tUpto + 1))
                        # Remove care workers
                        if Levels[tUpto + 1] < Levels[tUpto]:
                            for _ in range(Levels[tUpto + 1], Levels[tUpto]):
                                heapq.heappop(Agents)
                        # Move on
                        tUpto += 1

                # Go to the night shift once the next job must
                # start during the night shift
                elif not NightShift:
                    if int(max(k[0], min(Agents)) // self.L) not in self.T:
                        NightShift = True
                        # Drop down to night shift levels
                        for _ in range(len(Agents) - self.MinWorkers):
                            heapq.heappop(Agents)


            # Get the next care worker off the queue
            Agent = heapq.heappop(Agents)
            # Compute the delay
            ThisDelay = max(0, Agent - k[0])
            # Update the objective value
            Delay[min(len(self.T) - 1, int(k[0] // self.L))] += ThisDelay
            # Get the next idle time of the care worker
            Agent = max(Agent, k[0]) + k[1]
            # Put the care worker back into the agent queue
            heapq.heappush(Agents, Agent)

        # Dont bother handling any requests in the last
        # time period if we are in the initialization step
        if init:
            for tUpto in self.T[tUpto + 1:]:
                self.CareStart[s, tUpto] = tuple(Agents)
                self.ReqsStart[s, tUpto] = len(self.Scenario[s])

        # Cache and return the performance
        self.Simulations[s, TupleLevels] = Delay
        self.TimeInSimulation += time.time() - SimStartTime
        return Delay
