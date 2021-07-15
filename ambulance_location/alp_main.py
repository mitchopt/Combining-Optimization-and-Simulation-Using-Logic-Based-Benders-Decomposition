# -*- coding: utf-8 -*-
from alp_optimize import Optimize
from alp_helpers import GenNetwork, WriteNetwork
from alp_helpers import GenScenario, SaveScenario, LoadScenario
from alp_settings import Settings
from pathlib import Path
# import sys


def main():

    # ID of instance
    # Used to build unique file names
    # Re-running with the same ID will overwrite results
    ID = 1

    # Generate initial cuts prior to optimization
    Inits = True

    # Use brute force to test the solution (only works for M = 1)
    # Careful with this if K is large and/or M1 ~= K/2
    # Quickly leads to a combinatorial explosion that will fill your RAM
    BruteForce = False

    # Directory to store instance files
    InstanceDir = Path(__file__).parent / "instances"
    InstanceDir.mkdir(exist_ok=True)

    # Directory to store scenarios
    ScenariosDir = InstanceDir / f"instance_{ID}"
    ScenariosDir.mkdir(exist_ok=True)

    # Network filepath
    NetworkDir = InstanceDir / "networks"
    NetworkDir.mkdir(exist_ok=True)

    # Directory to store results files
    ResultsDir = Path(__file__).parent / "results"
    ResultsDir.mkdir(exist_ok=True)
    ResultsFile = ResultsDir / \
        f"ID_{ID}__Inits_{Inits}_results.csv"

    # Retrieve the instance parameters from the settings file
    Data = Settings()

    # Get the network we are going to use
    NetworkFile = NetworkDir / f"network_{ID}.csv"
    if not NetworkFile.exists():
        G, P, Station, Hospital = GenNetwork(Data, None)
        WriteNetwork(G, P, Station, Hospital, NetworkFile)

    # Generate and save scenarios
    if len(list(ScenariosDir.iterdir())) < 0.1:
        Scenario = GenScenario(Data, G, None)
        SaveScenario(ScenariosDir, Scenario)

    # Get the scenarios we will use
    ScenarioUse = LoadScenario(ScenariosDir)

    # Pass to the solver
    Optimize(Data, NetworkFile, ScenarioUse, Inits, ResultsFile, BruteForce)


if __name__ == "__main__":
    main()
