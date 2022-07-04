# -*- coding: utf-8 -*-
from model_IP import nursing_IP
from model_benders import nursing_BendersDecomposition
from instance_generation import GenerateScenario
from nursing_simulation import Nursing_Simulation
from pathlib import Path


def main():
    Seed = None
    
    # Type A Parameters
    # Filename = "DataNursingHome.csv"
    # ArrivalRate = 20
    # AvLowDuration = 1.89
    # AvHighDuration = 9.28
    # LowProb = 0.8
    # H = 16
    # L = 60
    # T = range(H)
    # N = 8
    # ShiftLengths = [4, 8]
    # MinWorkers = 2
    # MaxWorkers = 20
    # MaxHours = 80
        
    # Type B Parameters
    DataFile = Path(__file__).parent / "DataNursingHome.csv"
    ArrivalRate = 20
    AvLowDuration = 1.89
    AvHighDuration = 9.28
    LowProb = 0.8
    H = 4
    L = 60
    T = range(H)
    N = 2
    ShiftLengths = [1, 2]
    MinWorkers = 1
    MaxWorkers = 10
    MaxHours = 15
    
    # Generate requests
    Scenario = GenerateScenario(DataFile, Seed, ArrivalRate, AvLowDuration,
                                AvHighDuration, LowProb, H)
    
    # Purge every 4th request and quater the start times
    Scenario = Scenario[::4]
    for i in range(len(Scenario)):
        Scenario[i] = (round(Scenario[i][0] / 4), Scenario[i][1])
        
    
    # Create a simulation
    SimEngine = Nursing_Simulation(T, L, MinWorkers, MaxWorkers, [Scenario])
    
    
    # Solve by Benders decomposition
    modelBD, YSol = nursing_BendersDecomposition(SimEngine, True, True, H, L,
                                          ShiftLengths, MinWorkers, 
                                          MaxWorkers, MaxHours, None)

    # Solve IP with solution given
    modelIPf, _ = nursing_IP(Scenario, H, N, T, L, ShiftLengths, 
                              MinWorkers, MaxWorkers, MaxHours, YSol)
    
    # Solve IP without solution given
    modelIP, _ = nursing_IP(Scenario, H, N, T, L, ShiftLengths, 
                              MinWorkers, MaxWorkers, MaxHours, None)
    
    
if __name__ == "__main__":
    main()

    
    
