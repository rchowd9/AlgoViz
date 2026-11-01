import json

import streamlit as st

from visualization import (
    SEARCH_ALGORITHMS,
    hill_climb_trace,
    parse_graph,
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
SORTING_ALGORITHMS = ("Bubble sort", "Insertion sort", "Selection sort", "Merge sort")


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

search_tab, sorting_tab, optimization_tab, tensorflow_tab = st.tabs(
    ["Graph search", "Sorting", "Hill climbing", "TensorFlow"]
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
            use_container_width=True,
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
        history = model.fit(x[:, tf.newaxis], y, epochs=epochs, verbose=0)
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
