# -*- coding: utf-8 -*-
import numpy as np
import csv


# Scenario generation for ACCA problem
def GenScenarios(Data, Seed):

    # Seed the rng
    if type(Seed) == int:
        np.random.seed(seed=Seed)

    # Probability distribution for arrival period
    ArriveChoice = [x * 0.01 for x in Data.Arrive]
    ArrLen = len(Data.Arrive)

    S = Data.S
    F = Data.F
    Passengers = Data.Passengers
    Start = Data.Start
    L = Data.L
    ServiceTime = Data.ServiceTime

    # Senario[s] is a list of arrivals
    Scenario = [[] for s in S]
    for s in S:
        for f in F:
            for j in range(Passengers[f]):

                # Arrival time
                Arr = (np.random.choice(ArrLen, 1, p=ArriveChoice)[0]
                       + Start[f]) * L \
                    + np.random.randint(0, L + 1)

                # Service time
                Ser = np.random.exponential(ServiceTime)

                # Generate a single arrival
                Scenario[s].append((Arr, f, Ser))

        # Sort by arrival time
        Scenario[s].sort()
    return Scenario


# Save a set of scenarios to a directory
# Each scenario becomes a single file
def Save(Directory, Scenario):
    for s in range(len(Scenario)):
        with open(Directory / f"scenario_{s}.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            for row in Scenario[s]:
                writer.writerow(row)


# Load a set of scenarios from directory
# Each scenario is a single file
def Load(Directory):
    Scenario = []
    for path in Directory.iterdir():
        Scenario.append(list())
        with open(path, newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                Scenario[-1].append((float(row[0]), int(row[1]),
                                     float(row[2])))
    return Scenario


# Update the mean of a set of values without recalculating it from scratch
# This function is a remnant of something I tried. The idea was to load
# the average size of previous Benders cuts when generating the neighborhood
# instead of starting from {t} every time. This was not used in the end.
def RollAverage(NumPrevValues, PrevAv, NewValue):
    return PrevAv + (NewValue - PrevAv)/(NumPrevValues + 1)
