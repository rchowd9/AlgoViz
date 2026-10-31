import json
import math

import streamlit as st

from searching import (
    aStarSearch,
    breadthFirstSearch,
    depthLimitedSearch,
    depthFirstSearch,
    graphSearch,
    iterativeDeepeningAStarSearch,
    iterativeDeepeningSearch,
    treeSearch,
    uniformCostSearch,
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
ALGORITHMS = {
    "Breadth-first search": "bfs",
    "Depth-first search": "dfs",
    "Tree search": "tree",
    "Graph search": "graph",
    "Uniform-cost search": "ucs",
    "Depth-limited search": "dls",
    "Iterative deepening search": "ids",
    "A* search": "astar",
    "Iterative deepening A* search": "ida",
}


def parse_graph(graph_text):
    try:
        raw_graph = json.loads(graph_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Graph JSON is invalid: {error.msg}.") from error

    if not isinstance(raw_graph, dict) or not raw_graph:
        raise ValueError("The graph must be a non-empty JSON object.")

    graph = {}
    for node, neighbors in raw_graph.items():
        if not isinstance(node, str) or not isinstance(neighbors, dict):
            raise ValueError("Each node must map to an object of neighbor names and edge costs.")
        graph[node] = {}
        for neighbor, cost in neighbors.items():
            if not isinstance(neighbor, str) or not isinstance(cost, (int, float)):
                raise ValueError("Neighbor names must be strings and edge costs must be numbers.")
            if isinstance(cost, bool) or not math.isfinite(cost) or cost < 0:
                raise ValueError("Edge costs must be finite, non-negative numbers.")
            graph[node][neighbor] = cost

    for neighbors in graph.values():
        for neighbor in neighbors:
            if neighbor not in graph:
                raise ValueError(f"Neighbor {neighbor!r} is not defined as a graph node.")

    return graph


st.set_page_config(page_title="Algorithm Explorer", page_icon="🔎")
st.title("Search Algorithm Explorer")
st.write("Explore how search algorithms traverse a weighted graph.")

with st.sidebar:
    algorithm_name = st.selectbox("Algorithm", list(ALGORITHMS))
    graph_text = st.text_area(
        "Graph (JSON)",
        value=json.dumps(DEFAULT_GRAPH, indent=2),
        height=300,
        help="Use a JSON object where each node maps to its neighbors and edge costs.",
    )

try:
    weighted_graph = parse_graph(graph_text)
except ValueError as error:
    st.error(str(error))
    st.stop()

algorithm = ALGORITHMS[algorithm_name]
nodes = list(weighted_graph)
start = st.selectbox("Starting node", nodes)
depth = None

if algorithm in {"dls", "ids", "ida"}:
    depth = st.number_input("Maximum depth", min_value=0, max_value=100, value=3)

heuristic = None
if algorithm in {"astar", "ida"}:
    heuristic_text = st.text_area(
        "Heuristic values (JSON)",
        value=json.dumps(DEFAULT_HEURISTIC, indent=2),
        help="Provide a non-negative numeric heuristic for every graph node.",
    )
    try:
        heuristic = json.loads(heuristic_text)
        if not isinstance(heuristic, dict) or any(
            node not in heuristic
            or isinstance(heuristic[node], bool)
            or not isinstance(heuristic[node], (int, float))
            or not math.isfinite(heuristic[node])
            or heuristic[node] < 0
            for node in nodes
        ):
            raise ValueError("Provide a finite, non-negative heuristic for every graph node.")
    except (json.JSONDecodeError, ValueError) as error:
        st.error(f"Invalid heuristic: {error}")
        st.stop()

if st.button("Run search", type="primary"):
    adjacency_graph = {node: list(neighbors) for node, neighbors in weighted_graph.items()}

    if algorithm == "bfs":
        result = breadthFirstSearch(adjacency_graph, start)
    elif algorithm == "dfs":
        result = depthFirstSearch(adjacency_graph, start)
    elif algorithm == "tree":
        result = treeSearch(adjacency_graph, start)
    elif algorithm == "graph":
        result = graphSearch(adjacency_graph, start)
    elif algorithm == "ucs":
        result = uniformCostSearch(weighted_graph, start)
    elif algorithm == "dls":
        result = depthLimitedSearch(adjacency_graph, start, depth)
    elif algorithm == "ids":
        result = iterativeDeepeningSearch(adjacency_graph, start, depth)
    elif algorithm == "astar":
        result = aStarSearch(weighted_graph, start, heuristic)
    else:
        result = iterativeDeepeningAStarSearch(weighted_graph, start, heuristic, depth)

    st.subheader("Traversal result")
    st.write(result)
    st.caption(
        "This displays the order nodes are explored (and costs for weighted searches); "
        "it does not reconstruct a route to a goal."
    )
