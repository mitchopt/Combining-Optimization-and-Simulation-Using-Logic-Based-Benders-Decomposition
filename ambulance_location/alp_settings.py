# -*- coding: utf-8 -*-
import math


# Settings for ambulance location problem
class Settings():
    def __init__(self):

        # Samples
        self.NumSamples = 100

        # (*) Set of scenarios
        self.S = range(self.NumSamples)

        # Response time target
        self.Delta = 9

        # Max ambulances at a station
        self.M = 1

        # Total ambulances
        self.M1 = 30

        # Number of nodes
        self.NumNodes = 100

        # Number of stations
        self.NumStations = 40

        # Number of hospitals
        self.NumHosp = 10

        # Time period
        self.L = 60

        # Time periods
        self.H = 12

        # Average leave delay (195 seconds or 3.25 minutes)
        self.AvDelay = 3.25

        # Standard deviation (96 seconds or 1.6 minutes)
        self.SDDelay = 1.6

        # Average on-scene time (10 minutes)
        self.AvSceneTime = 10

        # Probability that a call needs a hospital
        self.ReqHospProb = 0.2

        # Average time at a hospital
        self.AvHospTime = 3

        # (*) Mean of lognormal delay
        self.Mu = 2*math.log(self.AvDelay) - \
            math.log(self.AvDelay**2 + self.SDDelay**2)/2

        # (*) Sigma for lognormal delay
        self.Sigma = math.sqrt(-2 * math.log(self.AvDelay) +
                               math.log(self.AvDelay**2 + self.SDDelay**2))
        # Rate factors
        self.Rate = (360, 10)

        # Size of city grid
        self.Size = 40

        # Minimum distance between nodes
        self.MinDist = 2

        # Maximum degree of nodes
        self.MaxDeg = 5

    # Update the attributes with keyword arguments
    # No error checking implemented
    def SetAttribute(self, **kwargs):
        for key, val in kwargs.items():
            if hasattr(self, key):
                self.__dict__[key] = val
