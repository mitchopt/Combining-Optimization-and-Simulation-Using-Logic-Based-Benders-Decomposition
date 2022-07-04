import numpy as np


def GenerateScenarios(NumScenarios, ServiceTime, Passengers, Start, Arrive, L):

    # Set of flights
    F = range(len(Passengers)) 
    
    # Set of arrivals
    Scenarios = [[] for s in range(NumScenarios)]
    S = range(len(Scenarios))
        
    # Get the arrival and service time
    # of each passenger on each flight
    # Scenario[p] = (arrival time, flight, service time)
    for s in S:
        for f in F:
            for p in range(Passengers[f]):
                
                Scenarios[s].append((float(L * (np.random.choice(
                    len(Arrive), 1, p = [i * 0.01 for i in Arrive])[0]
                    + Start[f]) + np.random.randint(0, L + 1)), 
                    f, float(np.random.exponential(ServiceTime))))
                
        Scenarios[s].sort()
            
    # Sort and return
    return Scenarios