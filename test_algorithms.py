import unittest
from unittest.mock import patch

from algorithms import (
    alphaBetaPruning,
    dijikstra,
    euclideanAlgorithm,
    geneticAlgorithm,
    kNearestNeighbors,
    miniMaxAlgorithm,
    simulatedAnnealing,
)


class GameState:
    def __init__(self, score=None, children=None):
        self.score = score
        self.children = children or []

    def get_children(self):
        return self.children


class IntegerState:
    def __init__(self, value):
        self.value = value

    def get_neighbors(self):
        return [IntegerState(self.value + 1)]


class AdditionalAlgorithmTests(unittest.TestCase):
    def test_genetic_algorithm_evolves_binary_string_and_records_trace(self):
        population = ["00", "00", "00", "00"]
        trace = []
        with patch("algorithms.random.random", return_value=0), patch(
            "algorithms.random.randrange", return_value=0
        ):
            result = geneticAlgorithm(
                population,
                lambda individual: individual.count("1"),
                1,
                1,
                trace=trace,
            )

        self.assertEqual(result, "10")
        self.assertEqual(population, ["00", "00", "00", "00"])
        self.assertEqual([step["generation"] for step in trace], [0, 1])
        self.assertEqual(trace[-1]["best_fitness"], 1)

    def test_genetic_algorithm_handles_single_member_and_zero_generations(self):
        self.assertEqual(
            geneticAlgorithm(["001"], lambda individual: individual.count("1"), 0, 0),
            "001",
        )

    def test_genetic_algorithm_validates_parameters(self):
        with self.assertRaisesRegex(ValueError, "population"):
            geneticAlgorithm([], lambda individual: 0, 1, 0.1)
        with self.assertRaisesRegex(ValueError, "mutation rate"):
            geneticAlgorithm(["0"], lambda individual: 0, 1, 1.1)

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

    def test_minimax_trace_records_leaf_evaluations_and_propagated_scores(self):
        tree = GameState(
            children=[
                GameState(children=[GameState(3), GameState(5)]),
                GameState(children=[GameState(2), GameState(9)]),
            ]
        )
        trace = []

        score = miniMaxAlgorithm(tree, 2, True, lambda state: state.score, trace=trace)

        self.assertEqual(score, 3)
        self.assertEqual(
            [step["score"] for step in trace if step["event"] == "Leaf evaluation"],
            [3, 5, 2, 9],
        )
        self.assertEqual(trace[-1]["event"], "Select maximum")
        self.assertEqual(trace[-1]["score"], score)

    def test_nearest_neighbor_trace_is_sorted_by_distance(self):
        data = [([0, 0], "near"), ([2, 2], "near"), ([9, 9], "far")]
        trace = []

        result = kNearestNeighbors(data, [1, 1], 2, trace=trace)

        self.assertEqual(result, "near")
        self.assertEqual([step["label"] for step in trace], ["near", "near", "far"])
        self.assertEqual(
            [step["distance"] for step in trace],
            sorted(step["distance"] for step in trace),
        )

    def test_simulated_annealing_trace_records_accepted_moves(self):
        trace = []
        with patch("algorithms.random.choice", side_effect=lambda neighbors: neighbors[-1]), patch(
            "algorithms.random.random", return_value=0
        ):
            result = simulatedAnnealing(
                IntegerState(0),
                lambda state: state.value,
                2,
                0.5,
                trace=trace,
            )

        self.assertEqual(result.value, 1)
        self.assertEqual(len(trace), 2)
        self.assertEqual(trace[1]["candidate_state"].value, 1)
        self.assertTrue(trace[1]["accepted"])

    def test_simulated_annealing_trace_records_rejected_moves(self):
        trace = []
        with patch("algorithms.random.choice", side_effect=lambda neighbors: neighbors[-1]), patch(
            "algorithms.random.random", return_value=0.9
        ):
            result = simulatedAnnealing(
                IntegerState(1),
                lambda state: -(state.value ** 2),
                2,
                0.5,
                trace=trace,
            )

        self.assertEqual(result.value, 1)
        self.assertFalse(trace[1]["accepted"])
        self.assertEqual(trace[1]["result_state"].value, 1)


if __name__ == "__main__":
    unittest.main()
