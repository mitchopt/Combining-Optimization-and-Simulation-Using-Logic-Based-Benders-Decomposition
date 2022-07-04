from instance_generation import GenerateScenarios
from airport_simulation import Airport_Simulation
from model_benders import airport_BendersDecomposition
from pathlib import Path
import csv
import sys


def main():
    
    
    # Parameters
    DeskCost = 40  # Cost of opening a desk for one time period
    QueueCost = 40  # Cost per time period for queuing time
    MaxDesks = 20 # Maximum desks which can be open at a time
    ServiceTime = 2  # Average service time
    # V  The number of passengers on each flight
    Passengers = [150, 210, 240, 180, 270, 150, 210, 300, 180, 270]
    # V  The start of the arrival periods for each flight
    Start = [0, 2, 4, 4, 6, 8, 10, 12, 12, 14]
    # V  The distribution of arrivals over each arrival period
    Arrive = [5, 10, 20, 30, 20, 15, 0]
    H = max(Start) + len(Arrive)  # (*) The number of time periods
    T = range(H)  # The set of time periods
    L = 30  # The length of each time period
    
    
    # Scenario generation
    NumScenarios = 1
    Scenarios = GenerateScenarios(NumScenarios, ServiceTime, Passengers, 
                                  Start, Arrive, L)
    

    # Use initial cuts?
    Inits = True
    
    
    # Create a simulation
    SimEngine = Airport_Simulation(T, L, MaxDesks, Scenarios)
    
    
    # Solve using Benders decomposition
    modelBD, YSol = airport_BendersDecomposition(
        SimEngine, Inits, H, L, Start, Arrive, MaxDesks, DeskCost, QueueCost)
    
    
if __name__=="__main__":
    main()


