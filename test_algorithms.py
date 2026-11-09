import unittest

from algorithms import (
    alphaBetaPruning,
    dijikstra,
    euclideanAlgorithm,
    kNearestNeighbors,
    miniMaxAlgorithm,
)


class GameState:
    def __init__(self, score=None, children=None):
        self.score = score
        self.children = children or []

    def get_children(self):
        return self.children


class AdditionalAlgorithmTests(unittest.TestCase):
    def test_dijkstra_returns_shortest_distances(self):
        graph = {
            "A": {"B": 4, "C": 1},
            "B": {"D": 1},
            "C": {"B": 1, "D": 8},
            "D": {},
        }
        self.assertEqual(dict(dijikstra(graph, "A")), {"A": 0, "C": 1, "B": 2, "D": 3})

    def test_nearest_neighbor_classifiers(self):
        data = [([0, 0], "near"), ([2, 2], "near"), ([9, 9], "far")]
        self.assertEqual(kNearestNeighbors(data, [1, 1], 2), "near")
        self.assertEqual(euclideanAlgorithm(data, [8, 8]), "far")

    def test_alpha_beta_matches_minimax(self):
        tree = GameState(
            children=[
                GameState(children=[GameState(3), GameState(5)]),
                GameState(children=[GameState(2), GameState(9)]),
            ]
        )
        evaluate = lambda state: state.score
        self.assertEqual(miniMaxAlgorithm(tree, 2, True, evaluate), 3)
        self.assertEqual(
            alphaBetaPruning(tree, 2, float("-inf"), float("inf"), True, evaluate),
            3,
        )


if __name__ == "__main__":
    unittest.main()
