from nhss_optimize import Optimize
from nhss_helpers import GenScenario, Save, Load
from nhss_settings import Settings
from pathlib import Path


def main():

    # ID of instance
    # Used to build unique file names
    # Re-running with the same ID will overwrite results
    ID = 1

    # Generate initial cuts prior to optimization
    Inits = True

    # Use the greedy heuristic prior to optimization
    Heuristic = True

    # Directory to store instance files
    InstanceDir = Path(__file__).parent / "instances"
    InstanceDir.mkdir(exist_ok=True)

    # Directory to store scenarios
    ScenariosDir = InstanceDir / f"instance_{ID}"
    ScenariosDir.mkdir(exist_ok=True)

    # Directory to store results files
    ResultsDir = Path(__file__).parent / "results"
    ResultsDir.mkdir(exist_ok=True)
    ResultsFile = ResultsDir / \
        f"ID_{ID}__Inits_{Inits}__Heur_{Heuristic}_results.csv"

    # Directory to store optimal solutions
    SolutionDir = Path(__file__).parent / "solutions"
    SolutionDir.mkdir(exist_ok=True)
    SolutionFile = SolutionDir / \
        f"ID_{ID}__Inits_{Inits}__Heu_{Heuristic}_solution.csv"

    # Directory to store the sizes of the Benders cuts generated
    SizesDir = Path(__file__).parent / "cut_sizes"
    SizesDir.mkdir(exist_ok=True)
    SizesFile = SizesDir / \
        f"ID_{ID}__Inits_{Inits}__Heu_{Heuristic}_sizes.csv"

    # Retrieve the instance parameters from the settings file
    Data = Settings()

    # Load samples from directory if any exist
    # Otherwise genrerate new scenarios
    if any(ScenariosDir.iterdir()):
        Scenario = Load(ScenariosDir)
        ScenariosInFolder = len(list(ScenariosDir.iterdir()))
        print("Loaded", ScenariosInFolder, "scenarios from", ScenariosDir)
        assert ScenariosInFolder == Data.NumS, f"{ScenariosInFolder} " \
            f"scenarios in the directory does not match {Data.NumS} " \
            "scenarios in the settings file"

    else:

        # Seed to use
        Seed = None

        print(f"Generating {Data.NumS} scenarios")
        Days = range(Data.NumDays)
        S = range(Data.NumS)
        # Scenario[s][d] is the list of requests for day d of scenario s
        Scenario = {s: {d: GenScenario(Data, Seed) for d in Days} for s in S}
        Save(ScenariosDir, Scenario)

    # Pass job data to the solver
    Optimize(Data, Scenario, Inits, Heuristic,
             ResultsFile, SolutionFile, SizesFile)


if __name__ == "__main__":
    main()
