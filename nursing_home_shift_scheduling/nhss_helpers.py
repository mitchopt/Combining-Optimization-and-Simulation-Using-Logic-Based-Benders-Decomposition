# -*- coding: utf-8 -*-
import csv
import pandas as pd
import numpy as np


# Generate the set of shifts of lengths in Lengths, starting
def GenShifts(Lengths, H, L):  # every L minutes for H hours
    assert 60 % L == 0, "L must be a divisor of 60 ..."

    # Generate the set of possible start times for shifts
    StartTimes = [60*t + l*L for t in range(H) for l in range(round(60 / L))
                  if 60*t + l*L + 60*min(Lengths) <= 60*H + 0.001]
    Shifts = []
    for l in Lengths:
        for start in StartTimes:

            # Add the possible shifts of each length
            if start + 60*l <= 60*H + 0.001:
                Shifts.append([start, start + 60*l])

            else:  # If a shift of this length doesn't fit
                continue  # now then it wont fit any later

    # Return
    return Shifts


# Convert HH:MM:SS to minutes and subtract "start"
def minutes(start, time):  # minutes from the total
    t = time.split(':')
    return (int(t[0]) * 60 + int(t[1]) + int(t[2]) / 60) - start


# Generate the matrix A used to link shift types
def GenMatrix(Shifts, L, T):  # to care worker levels

    # Most entries are zero
    A = [[0 for shift in Shifts] for t in T]
    for t in T:
        for (g, shift) in enumerate(Shifts):

            # Change A[t][shift] to 1 if shift type "shift"
            ttick = t * L + 0.01  # overlaps time period t
            if shift[0] <= ttick and ttick <= shift[1]:
                A[t][g] = 1

    # Return
    return A


# Update the mean of a set of values without recalculating it from scratch
# This function is a remnant of something I tried. The idea was to load
# the average size of previous Benders cuts when generating the neighborhood
# instead of starting from {t} every time. This was not used in the end.
def RollAverage(NumPrevValues, PrevAv, NewValue):
    return PrevAv + (NewValue - PrevAv)/(NumPrevValues + 1)


# Generate a scenario for one day by loading scheduled requests from a file
# and generating unscheduled requests with the specified parameters
def GenScenario(Data, Seed):

    if type(Seed) == int:
        np.random.seed(seed=Seed)

    Filename = Data.Filename
    data = pd.read_csv(Filename)
    length = data.shape[0]
    start = minutes(0, data["Request"][0])

    Requests = []
    # Scheduled requests
    # Add each request with a fixed probability
    for row in range(length):
        if np.random.random() < Data.IncludeProb:
            Requests.append((minutes(start, data["Request"][row]),
                             np.random.exponential(
                                 data["Expected service time"][row])))

    ArrivalRate = Data.ArrivalRate
    AvLowDuration = Data.AvLowDuration
    AvHighDuration = Data.AvHighDuration
    LowProb = Data.LowProb
    TimeHorizon = Data.H * 60

    tUpto = 0  # Generate unscheduled requests until
    while True:  # we reach the end of the time horizon
        Interval = max(1, round(np.random.exponential(ArrivalRate)))
        NextTime = tUpto + Interval
        if NextTime >= TimeHorizon:
            break

        # Add an unscheduled request
        tUpto = NextTime  # and move on
        if np.random.random() < LowProb:  # Add a short unscheduled request
            Requests.append((NextTime,
                             max(1, np.random.exponential(AvLowDuration))))

        else:  # Add a long unscheduled request
            Requests.append((NextTime,
                             max(1, np.random.exponential(AvHighDuration))))

    # Sort the requests by arrival time and
    Requests.sort()  # then return the list
    return Requests


# Save scenarios to csv files
def Save(Directory, Scenario):
    for s in range(len(Scenario)):
        for d in range(len(Scenario[s])):
            with open(Directory / f"scenario_{s}__day_{d}.csv",
                      'w', newline='') as file:
                writer = csv.writer(file)
                for row in Scenario[s][d]:
                    writer.writerow(row)


# Load all scenarios in directory
# The parsing done by this function depends on the file having
# the name structure from the "Save" function above
def Load(Directory):
    Scenario = {}
    for path in Directory.iterdir():
        name = str(path)

        # Get scenario and day of this file
        s = int(name.split("scenario_")[1][:3].translate({95: None}))
        d = int(name.split("day_")[1][:3].translate({j: None for j in
                                                     [46, 95, 99, 115, 118]}))

        if s not in Scenario:
            Scenario[s] = {}
        if d not in Scenario[s]:
            Scenario[s][d] = list()

        # Load the file
        with open(path, newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                Scenario[s][d].append((float(row[0]), float(row[1])))

    return Scenario
