import json
import random

import streamlit as st

from algorithms import (
    alphaBetaPruning,
    binaryGCDAlgorithm,
    dijikstra,
    euclideanAlgorithm,
    geneticAlgorithm,
    gradientDescent,
    kNearestNeighbors,
    miniMaxAlgorithm,
    simulatedAnnealing,
)
from visualization import (
    ARRAY_SEARCH_ALGORITHMS,
    SEARCH_ALGORITHMS,
    hill_climb_trace,
    parse_graph,
    search_array,
    search_graph,
    sorting_trace,
)


DEFAULT_GRAPH = {
    "A": {"B": 2, "C": 4},
    "B": {"A": 2, "D": 5, "E": 1},
    "C": {"A": 4, "F": 3},
    "D": {"B": 5, "G": 2},
    "E": {"B": 1, "G": 4},
    "F": {"C": 3, "G": 1},
    "G": {"D": 2, "E": 4, "F": 1},
}
DEFAULT_HEURISTIC = {"A": 6, "B": 4, "C": 3, "D": 2, "E": 3, "F": 1, "G": 0}
SORTING_ALGORITHMS = (
    "Bubble sort",
    "Insertion sort",
    "Selection sort",
    "Merge sort",
    "Shell sort",
    "Heap sort",
    "Counting sort",
    "Radix sort",
    "Pigeonhole sort",
    "Bucket sort",
)


class _IntegerState:
    def __init__(self, value, minimum=-10, maximum=10):
        self.value = value
        self.minimum = minimum
        self.maximum = maximum

    def get_neighbors(self):
        return [
            _IntegerState(value, self.minimum, self.maximum)
            for value in (self.value - 1, self.value + 1)
            if self.minimum <= value <= self.maximum
        ]


class _GameState:
    def __init__(self, score=None, children=None):
        self.score = score
        self.children = children or []

    def get_children(self):
        return self.children


def graphviz_source(graph, start, goal, order, path):
    visited = set(order)
    path_nodes = set(path or [])
    path_edges = set(zip(path or [], (path or [])[1:]))
    lines = [
        "digraph Search {",
        "  graph [rankdir=LR, bgcolor=\"transparent\"];",
        "  node [style=filled, fontname=\"Helvetica\"];",
        "  edge [fontname=\"Helvetica\", color=\"#a0aec0\"];",
    ]
    for node in graph:
        color = "#f6ad55" if node in path_nodes else "#90cdf4" if node in visited else "#e2e8f0"
        if node == start:
            color = "#68d391"
        if node == goal:
            color = "#fc8181" if node not in path_nodes else "#68d391"
        lines.append(f"  {json.dumps(node)} [label={json.dumps(node)}, fillcolor=\"{color}\"];")
    for node, neighbors in graph.items():
        for neighbor, cost in neighbors.items():
            edge_color = "#38a169" if (node, neighbor) in path_edges else "#a0aec0"
            penwidth = "3" if (node, neighbor) in path_edges else "1"
            lines.append(
                f"  {json.dumps(node)} -> {json.dumps(neighbor)} "
                f"[label={json.dumps(str(cost))}, color=\"{edge_color}\", penwidth=\"{penwidth}\"];"
            )
    lines.append("}")
    return "\n".join(lines)


st.set_page_config(page_title="Algorithm Explorer", page_icon="🔎", layout="wide")
st.title("Algorithm Explorer")
st.caption("Explore graph search, sorting, local optimization, and a TensorFlow neural network.")

search_tab, array_search_tab, sorting_tab, algorithms_tab, optimization_tab, more_algorithms_tab, tensorflow_tab = st.tabs(
    [
        "Graph search",
        "Array search",
        "Sorting",
        "Numeric algorithms",
        "Hill climbing",
        "More algorithms",
        "TensorFlow",
    ]
)

with search_tab:
    st.subheader("Visualize a search through a weighted graph")
    with st.sidebar:
        st.header("Graph search")
        graph_text = st.text_area(
            "Graph (JSON)",
            value=json.dumps(DEFAULT_GRAPH, indent=2),
            height=280,
            help="Each node maps to neighboring nodes and non-negative edge costs.",
        )

    try:
        weighted_graph = parse_graph(graph_text)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    nodes = list(weighted_graph)
    default_algorithm = SEARCH_ALGORITHMS.index("A* search")
    with st.form("search-controls"):
        first, second, third = st.columns(3)
        algorithm = first.selectbox("Search method", SEARCH_ALGORITHMS, index=default_algorithm)
        start = second.selectbox("Start node", nodes)
        goal = third.selectbox("Goal node", nodes, index=len(nodes) - 1)
        max_depth = st.number_input("Maximum depth (limited searches)", min_value=0, max_value=100, value=10)

        heuristic = None
        if algorithm in {
            "Greedy best-first search",
            "A* search",
            "Iterative deepening A* search",
        }:
            initial_heuristic = {node: DEFAULT_HEURISTIC.get(node, 0) for node in nodes}
            heuristic_text = st.text_area(
                "Heuristic values (JSON)",
                value=json.dumps(initial_heuristic, indent=2),
                help="Provide a finite, non-negative value for every node.",
            )
        else:
            heuristic_text = ""
        submitted = st.form_submit_button("Run search", type="primary")

    if submitted:
        try:
            heuristic = json.loads(heuristic_text) if heuristic_text else None
            result = search_graph(
                weighted_graph,
                start,
                goal,
                algorithm,
                heuristic=heuristic,
                max_depth=int(max_depth),
            )
            st.session_state["search_result"] = result
            st.session_state["search_settings"] = (algorithm, start, goal, weighted_graph)
        except (json.JSONDecodeError, ValueError) as error:
            st.error(f"Could not run search: {error}")

    saved_settings = st.session_state.get("search_settings")
    if saved_settings and saved_settings[3] == weighted_graph:
        algorithm, start, goal, _ = saved_settings
        result = st.session_state["search_result"]
        first, second, third = st.columns(3)
        first.metric("Nodes expanded", result["expanded"])
        second.metric("Path cost", result["cost"] if result["cost"] is not None else "No path")
        third.metric("Path length", len(result["path"]) if result["path"] else 0)
        st.graphviz_chart(
            graphviz_source(
                weighted_graph, start, goal, result["order"], result["path"]
            ),
            width="stretch",
        )
        st.write("**Expansion order:**", " → ".join(result["order"]) or "No nodes expanded")
        st.write(
            "**Path:**",
            " → ".join(result["path"]) if result["path"] else "No path found within the search limits.",
        )
        st.caption(
            "Green marks the start, red the goal, blue expanded nodes, orange route nodes, "
            "and green edges the final route."
        )
    elif saved_settings:
        st.info("The graph has changed. Run the search again to refresh the visualization.")
    else:
        st.info("Choose a search method and run it to see the traversal and route.")

with array_search_tab:
    st.subheader("Find a value in a sequence")
    with st.form("array-search-controls"):
        first, second, third = st.columns(3)
        array_search_algorithm = first.selectbox("Search method", ARRAY_SEARCH_ALGORITHMS)
        array_text = second.text_input("Numbers (comma-separated)", "8, 3, 5, 1, 9, 2")
        target_text = third.text_input("Target", "5")
        submitted_array_search = st.form_submit_button("Search", type="primary")

    if submitted_array_search:
        try:
            values = [int(value.strip()) for value in array_text.split(",") if value.strip()]
            target = int(target_text.strip())
            index = search_array(values, target, array_search_algorithm)
            st.session_state["array_search_result"] = {
                "values": values,
                "target": target,
                "index": index,
                "algorithm": array_search_algorithm,
            }
        except ValueError as error:
            st.error(f"Could not run search: {error}")

    array_result = st.session_state.get("array_search_result")
    if array_result:
        if array_result["index"] >= 0:
            st.success(f"Found {array_result['target']} at index {array_result['index']}.")
        else:
            st.info(f"{array_result['target']} was not found.")
        st.dataframe(
            [
                {"Index": index, "Value": value, "Match": value == array_result["target"]}
                for index, value in enumerate(array_result["values"])
            ],
            hide_index=True,
            use_container_width=True,
        )
        if array_result["algorithm"] != "Linear search":
            st.caption("Binary, ternary, and Fibonacci search require ascending input.")
    else:
        st.info("Choose a search method, enter a sequence and target, then run the search.")

with sorting_tab:
    st.subheader("Watch a sorting algorithm transform a sequence")
    with st.form("sorting-controls"):
        algorithm = st.selectbox("Sorting method", SORTING_ALGORITHMS)
        numbers_text = st.text_input("Numbers (comma-separated)", "8, 3, 5, 1, 9, 2")
        submitted_sort = st.form_submit_button("Create sorting trace", type="primary")

    if submitted_sort:
        try:
            values = [int(value.strip()) for value in numbers_text.split(",") if value.strip()]
            if len(values) > 50:
                raise ValueError("Use 50 numbers or fewer so each sorting step remains readable.")
            st.session_state["sorting_trace"] = sorting_trace(values, algorithm)
        except ValueError as error:
            st.error(f"Could not create sorting trace: {error}")

    steps = st.session_state.get("sorting_trace")
    if steps:
        step_number = st.slider("Step", 0, len(steps) - 1, len(steps) - 1)
        step = steps[step_number]
        st.caption(f"Step {step_number + 1} of {len(steps)} — {step['description']}")
        st.bar_chart(step["values"], height=320)
        st.write("**Sorted result:**", steps[-1]["values"])
    else:
        st.info("Enter a sequence and create a trace to inspect each sorting step.")

with algorithms_tab:
    st.subheader("Numeric algorithm demos")
    gcd_tab, gradient_tab = st.tabs(["Binary GCD", "Gradient descent"])
    with gcd_tab:
        st.write("Compute the greatest common divisor using Stein's binary GCD algorithm.")
        first, second = st.columns(2)
        gcd_a = first.number_input("First integer", value=48, step=1, key="gcd_a")
        gcd_b = second.number_input("Second integer", value=18, step=1, key="gcd_b")
        if st.button("Calculate GCD", type="primary"):
            result = binaryGCDAlgorithm(int(gcd_a), int(gcd_b))
            st.metric("Greatest common divisor", result)

    with gradient_tab:
        st.write("Minimize `f(x) = (x - 3)²` with gradient `2(x - 3)`.")
        initial_point = st.number_input("Starting point", value=-8.0, key="gradient_start")
        learning_rate = st.slider("Learning rate", 0.01, 0.5, 0.1, key="gradient_rate")
        iterations = st.slider("Iterations", 1, 100, 25, key="gradient_iterations")
        if st.button("Run gradient descent", type="primary"):
            point = gradientDescent(
                lambda coordinates: [2 * (coordinates[0] - 3)],
                [initial_point],
                learning_rate,
                iterations,
            )[0]
            st.metric("Final position", f"{point:.6f}")
            st.metric("Objective value", f"{(point - 3) ** 2:.6f}")

with optimization_tab:
    st.subheader("Hill climbing on a simple objective")
    st.write("The search maximizes `f(x) = -(x - 3)² + 9` by moving to a better neighboring integer.")
    initial = st.slider("Starting position", -10, 10, -8)
    if st.button("Run hill climbing", type="primary"):
        st.session_state["hill_climb_path"] = hill_climb_trace(initial)

    path = st.session_state.get("hill_climb_path")
    if path:
        st.write("**Visited positions:**", " → ".join(map(str, path)))
        st.line_chart(
            {"Objective score": [-(point - 3) ** 2 + 9 for point in path]},
            height=280,
        )
        st.success(f"Stopped at x = {path[-1]} with score {-(path[-1] - 3) ** 2 + 9}.")

with more_algorithms_tab:
    st.subheader("Explore additional algorithms")
    dijkstra_tab, annealing_tab, game_search_tab, knn_tab, genetic_tab = st.tabs(
        ["Dijkstra", "Simulated annealing", "Minimax", "Nearest neighbors", "Genetic algorithm"]
    )

    with dijkstra_tab:
        st.write("Find the shortest distance from a start node using the weighted graph above.")
        dijkstra_start = st.selectbox("Start node", list(weighted_graph), key="dijkstra_start")
        if st.button("Run Dijkstra", type="primary"):
            distances = dict(dijikstra(weighted_graph, dijkstra_start))
            st.session_state["dijkstra_result"] = [
                {
                    "Node": node,
                    "Distance": distances.get(node, "Unreachable"),
                }
                for node in weighted_graph
            ]
        if "dijkstra_result" in st.session_state:
            st.dataframe(st.session_state["dijkstra_result"], hide_index=True, use_container_width=True)

    with annealing_tab:
        st.write(
            "Maximize `f(x) = -(x - 3)² + 9`. Unlike hill climbing, simulated annealing "
            "can accept a worse move while the temperature is high."
        )
        annealing_start = st.slider("Starting position", -10, 10, -8, key="annealing_start")
        first, second = st.columns(2)
        temperature = first.slider("Initial temperature", 2, 100, 25, key="annealing_temperature")
        cooling_rate = second.slider("Cooling rate", 0.80, 0.99, 0.90, key="annealing_cooling")
        if st.button("Run simulated annealing", type="primary"):
            objective = lambda state: -(state.value - 3) ** 2 + 9
            annealing_trace = []
            final_state = simulatedAnnealing(
                _IntegerState(annealing_start),
                objective,
                temperature,
                cooling_rate,
                trace=annealing_trace,
            )
            st.session_state["annealing_result"] = {
                "position": final_state.value,
                "score": objective(final_state),
                "trace": annealing_trace,
            }
        annealing_result = st.session_state.get("annealing_result")
        if annealing_result:
            first, second = st.columns(2)
            first.metric("Final accepted position", annealing_result["position"])
            second.metric("Objective score", annealing_result["score"])
            st.caption("The final state may differ between runs because moves are sampled randomly.")
            trace = annealing_result["trace"]
            step_number = st.slider(
                "Annealing step",
                0,
                len(trace) - 1,
                len(trace) - 1,
                key="annealing_trace_step",
            )
            step = trace[step_number]
            if step["candidate_state"] is None:
                st.caption(
                    f"Step 1 of {len(trace)} — starting at x = {step['result_state'].value} "
                    f"with score {step['result_score']}."
                )
            else:
                decision = "accepted" if step["accepted"] else "rejected"
                st.caption(
                    f"Step {step_number + 1} of {len(trace)} — proposed "
                    f"x = {step['candidate_state'].value} (score {step['candidate_score']}) "
                    f"at temperature {step['temperature']:.2f}; {decision}."
                )
            visible_steps = trace[:step_number + 1]
            st.dataframe(
                [
                    {
                        "Step": index + 1,
                        "Temperature": item["temperature"],
                        "Current x": item["current_state"].value,
                        "Proposed x": (
                            item["candidate_state"].value
                            if item["candidate_state"] is not None
                            else None
                        ),
                        "Proposed score": item["candidate_score"],
                        "Decision": (
                            "Start"
                            if item["accepted"] is None
                            else "Accepted" if item["accepted"] else "Rejected"
                        ),
                        "Resulting x": item["result_state"].value,
                        "Resulting score": item["result_score"],
                    }
                    for index, item in enumerate(visible_steps)
                ],
                hide_index=True,
                use_container_width=True,
            )
            st.line_chart(
                {"Objective score": [item["result_score"] for item in visible_steps]},
                height=240,
            )

    with game_search_tab:
        st.write("Set eight terminal scores for a three-ply binary game tree.")
        default_scores = [3, 5, 2, 9, 12, 5, 23, 7]
        score_columns = st.columns(4)
        leaf_scores = [
            column.number_input(
                f"Leaf {index + 1}",
                value=score,
                step=1,
                key=f"minimax_leaf_{index}",
            )
            for index, (column, score) in enumerate(zip(score_columns * 2, default_scores))
        ]
        if st.button("Compare minimax methods", type="primary"):
            def build_tree(scores):
                if len(scores) == 1:
                    return _GameState(score=scores[0])
                midpoint = len(scores) // 2
                return _GameState(
                    children=[build_tree(scores[:midpoint]), build_tree(scores[midpoint:])]
                )

            def evaluate(state):
                return state.score

            tree = build_tree(leaf_scores)
            minimax_trace = []
            minimax_score = miniMaxAlgorithm(tree, 3, True, evaluate, trace=minimax_trace)
            pruned_score = alphaBetaPruning(
                tree,
                3,
                float("-inf"),
                float("inf"),
                True,
                evaluate,
            )
            st.session_state["game_search_result"] = (minimax_score, pruned_score)
            st.session_state["minimax_trace"] = minimax_trace
        game_search_result = st.session_state.get("game_search_result")
        if game_search_result:
            first, second = st.columns(2)
            first.metric("Minimax score", game_search_result[0])
            second.metric("Alpha-beta score", game_search_result[1])
            st.caption("Both methods return the same optimal score; alpha-beta can skip branches.")
            trace = st.session_state["minimax_trace"]
            step_number = st.slider(
                "Minimax step",
                0,
                len(trace) - 1,
                len(trace) - 1,
                key="minimax_trace_step",
            )
            step = trace[step_number]

            def format_node_path(node_path):
                if not node_path:
                    return "Root"
                return "Root → " + " → ".join(
                    "Left" if child_index == 0 else "Right"
                    for child_index in node_path
                )

            st.caption(
                f"Step {step_number + 1} of {len(trace)} — {step['event']} at "
                f"{format_node_path(step['node_path'])}: score {step['score']}."
            )
            st.dataframe(
                [
                    {
                        "Step": index + 1,
                        "Node": format_node_path(item["node_path"]),
                        "Depth": item["depth"],
                        "Turn": item["player"],
                        "Action": item["event"],
                        "Score": item["score"],
                    }
                    for index, item in enumerate(trace[:step_number + 1])
                ],
                hide_index=True,
                use_container_width=True,
            )

    with genetic_tab:
        st.write(
            "Maximize the number of 1s in a binary string. Selection, crossover, "
            "and mutation evolve a population toward the all-ones solution."
        )
        with st.form("genetic-algorithm-controls"):
            first, second = st.columns(2)
            genome_length = first.slider("Bit-string length", 4, 64, 16, key="genome_length")
            population_size = second.slider("Population size", 4, 100, 24, key="population_size")
            first, second = st.columns(2)
            generations = first.slider("Generations", 1, 100, 30, key="genetic_generations")
            mutation_rate = second.slider(
                "Mutation rate", 0.0, 1.0, 0.1, key="genetic_mutation_rate"
            )
            submitted_genetic = st.form_submit_button("Run genetic algorithm", type="primary")

        if submitted_genetic:
            initial_population = [
                "".join(random.choices("01", k=genome_length))
                for _ in range(population_size)
            ]
            genetic_trace = []
            best_individual = geneticAlgorithm(
                initial_population,
                lambda individual: individual.count("1"),
                generations,
                mutation_rate,
                trace=genetic_trace,
            )
            st.session_state["genetic_result"] = {
                "individual": best_individual,
                "fitness": best_individual.count("1"),
                "trace": genetic_trace,
            }

        genetic_result = st.session_state.get("genetic_result")
        if genetic_result:
            first, second = st.columns(2)
            first.metric("Best bit string", genetic_result["individual"])
            second.metric("Number of 1s", genetic_result["fitness"], delta=f"/ {genome_length}")
            st.line_chart(
                {
                    "Best fitness": [
                        step["best_fitness"] for step in genetic_result["trace"]
                    ],
                    "Average fitness": [
                        step["average_fitness"] for step in genetic_result["trace"]
                    ],
                },
                x="Generation",
                height=280,
            )

    with knn_tab:
        st.write("Classify a two-dimensional point using k-nearest neighbors.")
        with st.form("knn-controls"):
            training_text = st.text_area(
                "Training rows (x, y, label)",
                "1, 1, red\n2, 1, red\n4, 4, blue\n5, 4, blue",
                help="Enter one point and label per line, separated by commas.",
            )
            first, second, third = st.columns(3)
            query_x = first.number_input("Query x", value=3.5, key="knn_query_x")
            query_y = second.number_input("Query y", value=3.5, key="knn_query_y")
            neighbor_count = third.number_input(
                "k",
                min_value=1,
                max_value=50,
                value=3,
                step=1,
                key="knn_k",
            )
            submitted_knn = st.form_submit_button("Classify point", type="primary")

        if submitted_knn:
            try:
                training_data = []
                for line in training_text.splitlines():
                    if not line.strip():
                        continue
                    fields = [field.strip() for field in line.split(",")]
                    if len(fields) != 3 or not fields[2]:
                        raise ValueError("Each training row must contain x, y, and a non-empty label.")
                    training_data.append(
                        ((float(fields[0]), float(fields[1])), fields[2])
                    )
                if not training_data:
                    raise ValueError("Enter at least one training row.")
                if neighbor_count > len(training_data):
                    raise ValueError("k cannot exceed the number of training rows.")
                query_point = [query_x, query_y]
                neighbor_trace = []
                st.session_state["knn_result"] = {
                    "knn": kNearestNeighbors(
                        training_data,
                        query_point,
                        int(neighbor_count),
                        trace=neighbor_trace,
                    ),
                    "nearest": euclideanAlgorithm(training_data, query_point),
                    "query": query_point,
                    "k": int(neighbor_count),
                    "trace": neighbor_trace,
                }
            except ValueError as error:
                st.error(f"Could not classify point: {error}")

        knn_result = st.session_state.get("knn_result")
        if knn_result:
            first, second = st.columns(2)
            first.metric("k-NN class", knn_result["knn"])
            second.metric("Nearest-neighbor class", knn_result["nearest"])
            st.caption(f"Query point: ({knn_result['query'][0]}, {knn_result['query'][1]})")
            trace = knn_result["trace"]
            step_number = st.slider(
                "Neighbor step",
                1,
                len(trace),
                len(trace),
                key="knn_trace_step",
            )
            step = trace[step_number - 1]
            st.caption(
                f"Step {step_number} of {len(trace)} — rank {step['rank']} point "
                f"({step['point'][0]}, {step['point'][1]}) with label {step['label']!r} "
                f"is {step['distance']:.3f} units from the query."
            )
            st.dataframe(
                [
                    {
                        "Rank": item["rank"],
                        "Point": f"({item['point'][0]}, {item['point'][1]})",
                        "Label": item["label"],
                        "Distance": item["distance"],
                        "k-NN vote": (
                            "Included" if item["rank"] <= knn_result["k"] else "Outside k"
                        ),
                    }
                    for item in trace[:step_number]
                ],
                hide_index=True,
                use_container_width=True,
            )

with tensorflow_tab:
    st.subheader("Train a small TensorFlow regression model")
    st.write("A compact neural network learns the curve `sin(2x) + 0.15x` from synthetic data.")
    epochs = st.slider("Training epochs", min_value=5, max_value=100, value=25, step=5)

    if st.button("Train TensorFlow model", type="primary"):
        import tensorflow as tf

        tf.keras.utils.set_random_seed(7)
        x = tf.linspace(-2.0, 2.0, 128)
        y = tf.sin(2 * x) + 0.15 * x
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(1,)),
                tf.keras.layers.Dense(32, activation="tanh"),
                tf.keras.layers.Dense(32, activation="tanh"),
                tf.keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        history = model.fit(x[:, tf.newaxis], y[:, tf.newaxis], epochs=epochs, verbose=0)
        prediction = model.predict(x[:, tf.newaxis], verbose=0).reshape(-1).tolist()
        st.session_state["tensorflow_result"] = {
            "target": y.numpy().tolist(),
            "prediction": prediction,
            "loss": history.history["loss"],
        }

    tensorflow_result = st.session_state.get("tensorflow_result")
    if tensorflow_result:
        st.line_chart(
            {
                "Target": tensorflow_result["target"],
                "Model prediction": tensorflow_result["prediction"],
            },
            height=320,
        )
        st.line_chart({"Training loss (MSE)": tensorflow_result["loss"]}, height=180)
        st.metric("Final training loss", f"{tensorflow_result['loss'][-1]:.5f}")
    else:
        st.info("Start training to see the model prediction and loss curve.")

st.divider()
st.caption("The optional Flask JSON API is documented in the README and can run alongside this app.")
