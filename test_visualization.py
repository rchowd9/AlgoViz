import unittest

from api import app as flask_app
from visualization import parse_graph, search_graph, sorting_trace, hill_climb_trace


GRAPH = {
    "A": {"B": 4, "C": 1},
    "B": {"D": 1},
    "C": {"B": 1, "D": 8},
    "D": {},
}


class GraphSearchTests(unittest.TestCase):
    def test_astar_returns_lowest_cost_route(self):
        result = search_graph(
            GRAPH,
            "A",
            "D",
            "A* search",
            heuristic={"A": 0, "B": 0, "C": 0, "D": 0},
        )
        self.assertEqual(result["path"], ["A", "C", "B", "D"])
        self.assertEqual(result["cost"], 3)

    def test_bidirectional_search_returns_route_and_weighted_cost(self):
        result = search_graph(GRAPH, "A", "D", "Bidirectional search")
        self.assertEqual(result["path"][0], "A")
        self.assertEqual(result["path"][-1], "D")
        self.assertEqual(result["cost"], 5)

    def test_depth_limited_search_respects_limit(self):
        result = search_graph(GRAPH, "A", "D", "Depth-limited search", max_depth=1)
        self.assertFalse(result["found"])
        self.assertEqual(result["expanded"], 3)

    def test_graph_validation_rejects_undefined_neighbor(self):
        with self.assertRaisesRegex(ValueError, "not defined"):
            parse_graph({"A": {"missing": 1}})

    def test_unreachable_goal_is_reported(self):
        result = search_graph({"A": {}, "B": {}}, "A", "B", "Breadth-first search")
        self.assertFalse(result["found"])
        self.assertIsNone(result["path"])


class AlgorithmTraceTests(unittest.TestCase):
    def test_sorting_traces_end_with_sorted_values(self):
        for algorithm in ("Bubble sort", "Insertion sort", "Selection sort", "Merge sort"):
            with self.subTest(algorithm=algorithm):
                steps = sorting_trace([5, 1, 4, 2], algorithm)
                self.assertEqual(steps[-1]["values"], [1, 2, 4, 5])

    def test_hill_climbing_reaches_peak(self):
        self.assertEqual(hill_climb_trace(-8)[-1], 3)


class FlaskApiTests(unittest.TestCase):
    def setUp(self):
        flask_app.config["TESTING"] = True
        self.client = flask_app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "ok"})

    def test_search_endpoint_returns_route(self):
        response = self.client.post(
            "/api/search",
            json={
                "graph": GRAPH,
                "start": "A",
                "goal": "D",
                "algorithm": "Uniform-cost search",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["cost"], 3)

    def test_search_endpoint_reports_invalid_request(self):
        response = self.client.post("/api/search", json={"graph": {}})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())


if __name__ == "__main__":
    unittest.main()
