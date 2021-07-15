# -*- coding: utf-8 -*-
import itertools as it
import networkx as nx
import numpy as np
import csv

# For expanding polynomials
from sympy import symbols, expand


# Generate a network with stations and hospitals
def GenNetwork(Data, Seed):
    NumNodes = Data.NumNodes
    NumStations = Data.NumStations
    NumHospitals = Data.NumHosp
    MinDist = Data.MinDist
    MaxDeg = Data.MaxDeg
    Size = Data.Size

    # Seed the RNG
    if type(Seed) == int:
        np.random.seed(seed=Seed)

    # Graph object
    G = nx.Graph()
    G.add_nodes_from(range(NumNodes))

    # Generate points in the plane until we have enough
    P = {0: (Size*np.random.uniform(0, 1), Size*np.random.uniform(0, 1))}
    while len(P) < NumNodes:

        # Generate a new point
        p = (Size*np.random.uniform(0, 1), Size*np.random.uniform(0, 1))
        Norms = [np.linalg.norm((P[u][0]-p[0], P[u][1]-p[1]), 2) for u in P]
        if min(Norms) > MinDist:
            P[len(P)] = p

    # Add legal edges until the graph is connected
    Threshold = MinDist + 0.0001
    while not nx.is_connected(G):
        Threshold += 1
        NodesSorted = sorted(list(range(NumNodes)), key=lambda x: G.degree(x))
        for e in it.combinations(NodesSorted, 2):
            if e not in G.edges():
                if max(G.degree(e[0]), G.degree(e[1])) < MaxDeg:
                    d = np.linalg.norm((P[e[0]][0]-P[e[1]][0],
                                        P[e[0]][1]-P[e[1]][1]), 2)
                    if d < Threshold:
                        G.add_edge(e[0], e[1], weight=d)
                        if nx.is_connected(G):
                            break

    # Choose the stations and return the instance
    Stat = np.random.choice(range(NumNodes), NumStations, replace=False)
    Stations = {k: Stat[k] for k in range(NumStations)}
    Hospitals = {k: Stat[k] for k in range(NumHospitals)}
    return G, P, Stations, Hospitals


# Write the network to a file
def WriteNetwork(G, P, Station, Hospital, Writepath):
    """ Station = 1 if there is a station at the corresponding node;
    Station = 2 if there is a station and a hospital. The format is:
    -------------------
    Node, x, y, Station
    0,
    1,
    2,
       .
       .
       .

    Edges
    u, v, length
       .
       .
       .
    -------------------
    Saves to Writepath """

    with open(Writepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Nodes"])
        writer.writerow(["Node", "x", "y", "Station"])
        for p in range(len(P)):
            writer.writerow([p, P[p][0], P[p][0], int(p in Station.values()) +
                             int(p in Hospital.values())])
        writer.writerow(["Edges"])
        writer.writerow(["u", "v", "Weight"])
        for (u, v) in G.edges():
            writer.writerow([u, v, G[u][v]["weight"]])


# Load network from file
def LoadNetwork(Filepath):
    P = {}
    G = nx.Graph()
    Station = {}
    Hospital = {}

    part = 0
    with open(Filepath, newline='') as file:
        reader = csv.reader(file)
        next(reader)
        next(reader)
        for row in reader:
            # Change to edges
            if row[0] == "Edges":
                part = 1
                next(reader)
                continue

            # Read nodes
            if part == 0:
                G.add_node(int(row[0]))
                P[int(row[0])] = (float(row[1]), float(row[2]))
                if row[3] == "1":
                    Station[len(Station)] = int(row[0])
                if row[3] == "2":
                    Station[len(Station)] = int(row[0])
                    Hospital[len(Hospital)] = int(row[0])

            # Read edges
            if part == 1:
                G.add_edge(int(row[0]), int(row[1]), weight=float(row[2]))

    return G, P, Station, Hospital


# Generate scenarios with the specified parameters and seed
def GenScenario(Data, G, Seed):

    # Seed the RNG
    if type(Seed) == int:
        np.random.seed(seed=Seed)

    # Parameters
    S = Data.S
    V = range(Data.NumNodes)
    L = Data.L
    H = Data.H
    Rate = Data.Rate
    AvSceneTime = Data.AvSceneTime
    AvHospTime = Data.AvHospTime
    ReqHospProb = Data.ReqHospProb
    Mu = Data.Mu
    Sigma = Data.Sigma

    # Scenario[s] is a list of calls
    Scenario = {s: [] for s in S}
    for s in Data.S:
        for v in V:
            tUpTo = 0
            while tUpTo < L * H - 1:
                # Interval before the next call
                Interval = np.random.exponential(Rate[0] - Rate[1]*G.degree(v))
                # Decide if this call needs a hospital
                Hosp = np.random.uniform(0, 1) < ReqHospProb
                # Don't add calls after the time horion
                if tUpTo + Interval < L * H - 1:

                    Scenario[s].append([
                        # c[0]: Time of the call
                        round(tUpTo + Interval, 1),
                        # c[1]: Location
                        v,
                        # c[2]: Pre-travel delay
                        round(np.random.lognormal(mean=Mu, sigma=Sigma), 1),
                        # c[3]: On-scene time
                        round(np.random.exponential(AvSceneTime), 1),
                        # c[4]: Hospital or not
                        Hosp,
                        # c[5]: Hospital time
                        np.random.exponential(AvHospTime) * int(Hosp)
                        ])

                tUpTo += Interval
        Scenario[s].sort()
    return Scenario


# Save a list of scenarios to a directory
# Each scenario becomes its own csv file
def SaveScenario(Directory, Scenario):
    for s in range(len(Scenario)):
        with open(Directory / f"scenario_{s}.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            for row in Scenario[s]:
                writer.writerow(row)


# Load all scenarios from a directory
# Return a list of scenarios
def LoadScenario(Directory):
    Scenario = []
    for path in Directory.iterdir():
        Scenario.append(list())
        with open(path, newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                Bool = row[4] == "True"
                Scenario[-1].append((float(row[0]), int(row[1]),
                                     float(row[2]), float(row[3]), Bool,
                                     float(row[5])))
    return Scenario


# Calculate the number of feasible solutions
def CalcFeasible(KLen, M, M1):
    x = symbols('x')
    Polynomial = 0
    for i in range(M + 1):
        Polynomial = Polynomial + x**i
    Expression = expand(Polynomial**KLen)
    return Expression.coeff(x, M1)
