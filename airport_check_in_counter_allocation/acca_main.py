# -*- coding: utf-8 -*-
from acca_optimize import Optimize
from acca_helpers import GenScenarios, Save, Load
from acca_settings import Settings
from pathlib import Path


def main():

    # ID of instance
    # Used to build unique file names
    # Re-running with the same ID will overwrite results
    ID = 1

    # Queuing cost
    Q = 40

    # Benders cut
    # Method = 0  # No-good cut
    # Method = 1  # Monotonic cut
    # Method = 2  # Localized cut
    Method = 3  # Strengthened cut

    # Use initial cuts when solving
    Inits = True

    # Instances directory
    InstanceDir = Path(__file__).parent / "instances"
    InstanceDir.mkdir(exist_ok=True)

    # Scenario logs location
    ScenariosDir = InstanceDir / f"instance_{ID}"
    ScenariosDir.mkdir(exist_ok=True)

    # Results location
    ResultsDir = Path(__file__).parent / "results"
    ResultsDir.mkdir(exist_ok=True)
    ResultsFile = ResultsDir / \
        f"ID_{ID}__Q_{Q}__Method_{Method}__Inits_{Inits}_results.csv"

    # Optimal solutions location
    SolutionDir = Path(__file__).parent / "solutions"
    SolutionDir.mkdir(exist_ok=True)
    SolutionFile = SolutionDir / \
        f"ID_{ID}__Q_{Q}__Method_{Method}__Inits_{Inits}_solution.csv"

    # Create the instance object
    Data = Settings()

    # Updata the queue cost for this experiment
    Data.SetAttribute(QueueCost=Q)

    # Generate scenarios (if there aren't some already)
    # Be careful if you used SetAttribute to modify any
    #    parameters relevant to the scenario generation
    if len(list(ScenariosDir.iterdir())) < 0.1:
        Scenarios = GenScenarios(Data, None)
        Save(ScenariosDir, Scenarios)

    # Get the scenarios we will use
    ScenarioUse = Load(ScenariosDir)

    # Optimize the model (second arg is whether we use initial cuts)
    Optimize(Data, True, ScenarioUse, Method, Inits,
             ResultsFile, SolutionFile)


if __name__ == "__main__":
    main()
