# -*- coding: utf-8 -*-


class Settings():
    def __init__(self):

        # Cost of opening a desk for one time period
        self.DeskCost = 40

        # Cost per time period for queuing time
        self.QueueCost = 40

        # Maximum desks which can be open at a time
        self.MaxDesks = 20

        # The number of scenarios
        self.NumScenarios = 100

        # (*) The set of scenarios
        self.S = range(self.NumScenarios)

        # Average service time
        self.ServiceTime = 2

        # The number of passengers on each flight
        self.Passengers = [150, 210, 240, 180, 270, 150, 210, 300, 180, 270]

        # (*) The set of flights
        self.F = range(len(self.Passengers))

        # The start of the arrival periods for each flight
        self.Start = [0, 2, 4, 4, 6, 8, 10, 12, 12, 14]

        # The distribution of arrivals over each arrival period
        self.Arrive = [5, 10, 20, 30, 20, 15, 0]

        # (*) The number of time periods
        self.H = max(self.Start) + len(self.Arrive)

        # The length of each time period
        self.L = 30

    # Update the attributes with keyword arguments
    # No error checking implemented
    def SetAttribute(self, **kwargs):
        for key, val in kwargs.items():
            if hasattr(self, key):
                self.__dict__[key] = val
