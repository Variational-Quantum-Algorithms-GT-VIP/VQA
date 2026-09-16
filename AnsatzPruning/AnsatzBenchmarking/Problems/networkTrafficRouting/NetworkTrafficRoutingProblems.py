from ..base import ProblemSet
from .NetworkTrafficRoutingHamiltonian import buildNetworkTrafficRoutingHamiltonian
import networkx as nx
from qiskit.quantum_info import SparsePauliOp


# Each path is a list of edges
# Example path: [(0,1), (1,2)]


def problem1():
    G = nx.Graph()
    G.add_edge(0, 1, weight=5.0)
    G.add_edge(1, 2, weight=5.0)

    paths = [
        [(0, 1), (1, 2)],   # Path A
        [(0, 1)],           # Path B
    ]

    # Best: choose only path B -> cost = 5
    expectedAns = 5.0
    return (G, paths, expectedAns)


def problem2():
    G = nx.Graph()
    G.add_edge(0, 1, weight=3.0)

    paths = [
        [(0, 1)],
        [(0, 1)]
    ]

    # Choosing both causes congestion:
    # f = 2 → cost = 3 * 4 = 12
    # Best: choose one -> 3
    expectedAns = 3.0
    return (G, paths, expectedAns)


def problem3():
    G = nx.Graph()
    G.add_edge(0, 1, weight=2.0)
    G.add_edge(1, 2, weight=2.0)

    paths = [
        [(0, 1)],
        [(0, 1), (1, 2)],
        [(1, 2)]
    ]

    expectedAns = 2.0
    return (G, paths, expectedAns)


def problem4():
    G = nx.Graph()
    G.add_edge(0, 1, weight=4.0)
    G.add_edge(1, 2, weight=4.0)
    G.add_edge(0, 2, weight=1.0)

    paths = [
        [(0, 1), (1, 2)],  # heavy route
        [(0, 2)]           # light direct route
    ]

    expectedAns = 1.0
    return (G, paths, expectedAns)


def problem5():
    G = nx.Graph()
    G.add_edge(0, 1, weight=10.0)

    paths = [
        [(0, 1)],
        [(0, 1)],
        [(0, 1)]
    ]

    # Best: choose one only -> 10
    expectedAns = 10.0
    return (G, paths, expectedAns)

# Expanded Problem Set
def problem6():
    G = nx.Graph()
    #8 qubit example - K5 graph with equal weights
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(0, 2, weight=1.0)
    G.add_edge(0, 3, weight=1.0)
    G.add_edge(0, 4, weight=1.0)

    G.add_edge(1, 2, weight=1.0)
    G.add_edge(1, 3, weight=1.0)
    G.add_edge(1, 4, weight=1.0)

    G.add_edge(2, 3, weight=1.0)
    G.add_edge(2, 4, weight=1.0)

    G.add_edge(3, 4, weight=1.0)

    paths = [
        [(0, 4)],                           # Optimal direct Path A
        [(0, 1), (1, 4)],                   # Path B
        [(0, 2), (2, 4)],                   # Path B
        [(0, 3), (3, 4)],                   # Path B

        [(0, 1), (1, 2), (2, 4)],           # Path C
        [(0, 1), (1, 3), (3, 4)],           # Path C
        [(0, 2), (2, 3), (3, 4)],           # Path C

        [(0, 1), (1, 2), (2, 3), (3, 4)]    # Path D
    ]

    # Best: choose only path A -> cost = 1.0
    expectedAns = 1.0
    return (G, paths, expectedAns)

def problem7():
    G = nx.Graph()
    #8 qubit example - K5 graph with different weights
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(0, 2, weight=5.0)
    G.add_edge(0, 3, weight=5.0)
    G.add_edge(0, 4, weight=10.0)

    G.add_edge(1, 2, weight=1.0)
    G.add_edge(1, 3, weight=5.0)
    G.add_edge(1, 4, weight=10.0)

    G.add_edge(2, 3, weight=1.0)
    G.add_edge(2, 4, weight=10.0)

    G.add_edge(3, 4, weight=1.0)

    paths = [
        [(0, 4)],                           # Path A
        [(0, 1), (1, 4)],                   # Path B
        [(0, 2), (2, 4)],                   # Path B
        [(0, 3), (3, 4)],                   # Path B

        [(0, 1), (1, 2), (2, 4)],           # Path C
        [(0, 1), (1, 3), (3, 4)],           # Path C
        [(0, 2), (2, 3), (3, 4)],           # Path C

        [(0, 1), (1, 2), (2, 3), (3, 4)]    # Optimal Path D - constructed to be lowest cost
    ]

    # Best: choose path D -> cost = 4.0
    expectedAns = 4.0
    return (G, paths, expectedAns)

def problem8():
    G = nx.Graph()
    #16 qubit example - K6 graph with equal weights
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(0, 2, weight=1.0)
    G.add_edge(0, 3, weight=1.0)
    G.add_edge(0, 4, weight=1.0)
    G.add_edge(0, 5, weight=1.0)

    G.add_edge(1, 2, weight=1.0)
    G.add_edge(1, 3, weight=1.0)
    G.add_edge(1, 4, weight=1.0)
    G.add_edge(1, 5, weight=1.0)

    G.add_edge(2, 3, weight=1.0)
    G.add_edge(2, 4, weight=1.0)
    G.add_edge(2, 5, weight=1.0)

    G.add_edge(3, 4, weight=1.0)
    G.add_edge(3, 5, weight=1.0)

    G.add_edge(4, 5, weight=1.0)

    paths = [
        [(0, 5)],                                   # Optimal direct Path A
        [(0, 1), (1, 5)],                           # Path B
        [(0, 2), (2, 5)],                           # Path B
        [(0, 3), (3, 5)],                           # Path B
        [(0, 4), (4, 5)],                           # Path B
        [(0, 1), (1, 2), (2, 5)],                   # Path C
        [(0, 1), (1, 3), (3, 5)],                   # Path C
        [(0, 1), (1, 4), (4, 5)],                   # Path C
        [(0, 2), (2, 3), (3, 5)],                   # Path C
        [(0, 2), (2, 4), (4, 5)],                   # Path C
        [(0, 3), (3, 4), (4, 5)],                   # Path C
        [(0, 1), (1, 2), (2, 3), (3, 5)],           # Path D
        [(0, 1), (1, 2), (2, 4), (4, 5)],           # Path D
        [(0, 1), (1, 3), (3, 4), (4, 5)],           # Path D
        [(0, 2), (2, 3), (3, 4), (4, 5)],           # Path D
        #[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]    # Path E - Removed to reduce qubit count to 15
    ]

    # Best: choose only path A -> cost = 1.0
    expectedAns = 1.0
    return (G, paths, expectedAns)

def problem9():
    G = nx.Graph()
    #16 qubit example - K6 graph with different weights
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(0, 2, weight=5.0)
    G.add_edge(0, 3, weight=5.0)
    G.add_edge(0, 4, weight=5.0)
    G.add_edge(0, 5, weight=10.0)

    G.add_edge(1, 2, weight=1.0)
    G.add_edge(1, 3, weight=5.0)
    G.add_edge(1, 4, weight=5.0)
    G.add_edge(1, 5, weight=10.0)

    G.add_edge(2, 3, weight=1.0)
    G.add_edge(2, 4, weight=5.0)
    G.add_edge(2, 5, weight=10.0)

    G.add_edge(3, 4, weight=1.0)
    G.add_edge(3, 5, weight=10.0)

    G.add_edge(4, 5, weight=1.0)

    paths = [
        #[(0, 5)],                                   # Path A - Removed to reduce qubit count to 15
        [(0, 1), (1, 5)],                           # Path B
        [(0, 2), (2, 5)],                           # Path B
        [(0, 3), (3, 5)],                           # Path B
        [(0, 4), (4, 5)],                           # Path B
        [(0, 1), (1, 2), (2, 5)],                   # Path C
        [(0, 1), (1, 3), (3, 5)],                   # Path C
        [(0, 1), (1, 4), (4, 5)],                   # Path C
        [(0, 2), (2, 3), (3, 5)],                   # Path C
        [(0, 2), (2, 4), (4, 5)],                   # Path C
        [(0, 3), (3, 4), (4, 5)],                   # Path C
        [(0, 1), (1, 2), (2, 3), (3, 5)],           # Path D
        [(0, 1), (1, 2), (2, 4), (4, 5)],           # Path D
        [(0, 1), (1, 3), (3, 4), (4, 5)],           # Path D
        [(0, 2), (2, 3), (3, 4), (4, 5)],           # Path D
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]    # Optimal Path E - constructed to be lowest cost
    ]

    # Best: choose path E -> cost = 5.0
    expectedAns = 5.0
    return (G, paths, expectedAns)

def problem10():
    G = nx.Graph()
    #32 qubit example - K7 graph with equal weights
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(0, 2, weight=1.0)
    G.add_edge(0, 3, weight=1.0)
    G.add_edge(0, 4, weight=1.0)
    G.add_edge(0, 5, weight=1.0)
    G.add_edge(0, 6, weight=1.0)

    G.add_edge(1, 2, weight=1.0)
    G.add_edge(1, 3, weight=1.0)
    G.add_edge(1, 4, weight=1.0)
    G.add_edge(1, 5, weight=1.0)
    G.add_edge(1, 6, weight=1.0)

    G.add_edge(2, 3, weight=1.0)
    G.add_edge(2, 4, weight=1.0)
    G.add_edge(2, 5, weight=1.0)
    G.add_edge(2, 6, weight=1.0)

    G.add_edge(3, 4, weight=1.0)
    G.add_edge(3, 5, weight=1.0)
    G.add_edge(3, 6, weight=1.0)

    G.add_edge(4, 5, weight=1.0)
    G.add_edge(4, 6, weight=1.0)

    G.add_edge(5, 6, weight=1.0)

    paths = [
        [(0, 6)],                                           # Optimal direct Path A
        #[(0, 1), (1, 6)],                                   # Path B - Removed to reduce qubit count to 21
        #[(0, 2), (2, 6)],                                   # Path B
        #[(0, 3), (3, 6)],                                   # Path B
        #[(0, 4), (4, 6)],                                   # Path B
        #[(0, 5), (5, 6)],                                   # Path B
        [(0, 1), (1, 2), (2, 6)],                           # Path C
        [(0, 1), (1, 3), (3, 6)],                           # Path C
        [(0, 1), (1, 4), (4, 6)],                           # Path C
        [(0, 1), (1, 5), (5, 6)],                           # Path C
        [(0, 2), (2, 3), (3, 6)],                           # Path C
        [(0, 2), (2, 4), (4, 6)],                           # Path C
        [(0, 2), (2, 5), (5, 6)],                           # Path C
        [(0, 3), (3, 4), (4, 6)],                           # Path C
        [(0, 3), (3, 5), (5, 6)],                           # Path C
        [(0, 4), (4, 5), (5, 6)],                           # Path C
        [(0, 1), (1, 2), (2, 3), (3, 6)],                   # Path D
        [(0, 1), (1, 2), (2, 4), (4, 6)],                   # Path D
        [(0, 1), (1, 2), (2, 5), (5, 6)],                   # Path D
        [(0, 1), (1, 3), (3, 4), (4, 6)],                   # Path D
        [(0, 1), (1, 3), (3, 5), (5, 6)],                   # Path D
        [(0, 1), (1, 4), (4, 5), (5, 6)],                   # Path D
        [(0, 2), (2, 3), (3, 4), (4, 6)],                   # Path D
        [(0, 2), (2, 3), (3, 5), (5, 6)],                   # Path D
        [(0, 2), (2, 4), (4, 5), (5, 6)],                   # Path D
        [(0, 3), (3, 4), (4, 5), (5, 6)],                   # Path D
        #[(0, 1), (1, 2), (2, 3), (3, 4), (4, 6)],           # Path E
        #[(0, 1), (1, 2), (2, 3), (3, 5), (5, 6)],           # Path E
        #[(0, 1), (1, 2), (2, 4), (4, 5), (5, 6)],           # Path E
        #[(0, 1), (1, 3), (3, 4), (4, 5), (5, 6)],           # Path E
        #[(0, 2), (2, 3), (3, 4), (4, 5), (5, 6)],           # Path E
        #[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]    # Path F

    ]

    # Best: choose path A -> cost = 1.0
    expectedAns = 1.0
    return (G, paths, expectedAns)

class NetworkTrafficRoutingProblemSet(ProblemSet):

    def createProblemSets(self) -> list[tuple[SparsePauliOp, float]]:

        graphs = [
            problem1(),
            problem2(),
            problem3(),
            problem4(),
            problem5(),
            problem6(),
            problem7(),
            problem8(),
            problem9(),
            problem10()
        ]

        problems = [(None, None)] * len(graphs)

        for i, (graph, paths, ans) in enumerate(graphs):
            problems[i] = (
                buildNetworkTrafficRoutingHamiltonian(graph, paths),
                ans
            )

        return problems
