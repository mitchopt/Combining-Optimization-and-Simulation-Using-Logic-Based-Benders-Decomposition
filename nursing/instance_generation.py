# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from helper_functions import minutes


# Generate a single scenario for one day by loading scheduled requests from a file
# and generating unscheduled requests with the specified parameters
def GenerateScenario(Filename, Seed,
        ArrivalRate, AvLowDuration, AvHighDuration, LowProb, H):

    # Set seed
    if type(Seed) == int:
        np.random.seed(seed=Seed)


    # Load scheduled requests
    data = pd.read_csv(Filename)
    length = data.shape[0]
    start = minutes(0, data["Request"][0])

    # Set of requests
    Requests = []

    # Scheduled requests with durations sampled from
    # expected service time
    for row in range(length):
        Requests.append((round(minutes(start, data["Request"][row])),
                         round(max(1, np.random.exponential(
                             data["Expected service time"][row])))))


    tUpto = 0  # Generate unscheduled requests until
    while True:  # we reach the end of the time horizon
        Interval = max(1, round(np.random.exponential(ArrivalRate)))
        NextTime = tUpto + Interval
        if NextTime >= H * 60:
            break

        # Add a short unscheduled request
        tUpto = NextTime  # and move on
        if np.random.random() < LowProb:  # Add a short unscheduled request
            Requests.append((round(NextTime),
                             round(max(1, np.random.exponential(AvLowDuration)))))

        else:  # Add a long unscheduled request
            Requests.append((round(NextTime),
                             round(max(1, np.random.exponential(AvHighDuration)))))

    # Sort the requests by arrival time and
    Requests.sort()  # then return the list
    return Requests