from pathlib import Path
from nhss_helpers import GenShifts, GenMatrix


class Settings():

    def __init__(self):

        # Number of days
        self.NumDays = 1

        # Primary hours in a day
        self.H = 16

        # Start time for day shifts (HH:MM:SS) (will be converted)
        self.DayStart = "07:00:00"

        # Length of a time period
        self.L = 60

        # (*) The set of time periods for each day
        self.T = range(self.H * round(60 / self.L))

        # Set of shift lengths
        self.Lengths = [4, 8]

        # Min workers to be available at any time
        self.MinWorkers = 2  # including the night-shift

        # Max working hours
        self.MaxHours = 80 * self.NumDays

        # Max workers available at any time
        self.MaxWorkers = 20

        # (*) The set of shift types
        self.Shifts = GenShifts(self.Lengths, self.H, self.L)

        # (*) The matrix A linking shifts to levels
        self.A = GenMatrix(self.Shifts, self.L, self.T)

        # The number of scenarios
        self.NumS = 100

        # Data for scheduled requests
        self.Filename = "NHSS_scheduled_requests.csv"

        # Inclusion probability for scheduled requests
        self.IncludeProb = 1

        # Arrival rate of unscheduled requests
        self.ArrivalRate = 20

        # Average length of a short request
        self.AvLowDuration = 1.89

        # Average length of a long request
        self.AvHighDuration = 9.28

        # Probability of an unscheduled request being short
        self.LowProb = 0.8

    # Update the attributes with keyword arguments
    # No error checking implemented
    def SetAttribute(self, **kwargs):
        for key, val in kwargs.items():
            if hasattr(self, key):
                self.__dict__[key] = val
