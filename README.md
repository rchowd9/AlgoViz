# Algorithm Explorer

An interactive learning app for visualizing classic algorithms and search
strategies. The Streamlit UI includes:

- **Graph search:** breadth-first, depth-first, uniform-cost, greedy best-first,
  A*, depth-limited, iterative deepening, iterative deepening A*, and
  bidirectional search. Edit a weighted graph as JSON, choose start and goal
  nodes, and inspect the expansion order, route, and graph visualization.
- **Array search:** linear, binary, ternary, and Fibonacci search. Ordered
  searches validate that the input is ascending.
- **Sorting:** bubble, insertion, selection, merge, shell, heap, counting,
  radix, pigeonhole, and bucket sort using a sequence of up to 50 integers.
- **Numeric algorithms:** run binary GCD and gradient descent on an editable
  starting value.
- **Hill climbing:** follow a local optimization search on a one-dimensional
  objective.
- **Additional algorithms:** compare Dijkstra distances on the editable graph,
  run simulated annealing on the hill-climbing objective, compare minimax with
  alpha-beta pruning on a sample game tree, classify points with k-nearest
  neighbors, and evolve a binary string with a genetic algorithm. Step-by-step
  traces show annealing proposals and decisions, minimax evaluations and
  propagated scores, nearest-neighbor distances, and genetic-algorithm fitness.
- **TensorFlow:** train a small neural network on synthetic regression data and
  view its predictions and training-loss curve. TensorFlow is imported only
  when training is requested.
- **Flask JSON API:** run graph searches from another application or script.

## Run the Streamlit app

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

In the graph editor, each node maps to neighboring nodes and non-negative edge
costs:

```json
{
  "A": {"B": 2, "C": 4},
  "B": {"A": 2},
  "C": {"A": 4}
}
```

Heuristic-based searches require a finite, non-negative heuristic value for
every node. A* finds an optimal route when the heuristic is admissible (and
graph-search A* is guaranteed optimal with a consistent heuristic); greedy
best-first search is not guaranteed to find the lowest-cost route. Breadth-first
and bidirectional search minimize the number of edges, not the total weight.

## Run the Flask API

Run this in a separate terminal; the API and Streamlit app share the same search
implementation:

```bash
python -m flask --app api run --host 127.0.0.1 --port 5000
```

Health check: `GET http://127.0.0.1:5000/health`

Search: `POST http://127.0.0.1:5000/api/search`

```json
{
  "graph": {"A": {"B": 2}, "B": {}},
  "start": "A",
  "goal": "B",
  "algorithm": "Uniform-cost search"
}
```

For A* or greedy best-first search, include a `heuristic` object with a value
for every graph node. Depth-limited methods accept an optional `max_depth`
integer (default `20`). Successful responses include `order`, `path`, `cost`,
`found`, and `expanded`; invalid requests return HTTP 400 with an `error`
message.

## Deploy on Streamlit Community Cloud

1. Push the project to GitHub.
2. Create an app in Streamlit Community Cloud, select the repository, and set
   the main file to `app.py`.
3. Deploy. The platform installs Streamlit, TensorFlow, and Flask from
   `requirements.txt`.

The Streamlit deployment hosts the visual app. The Flask API is a separate
process and should be deployed as its own web service if it needs to be
available publicly.

## Run tests

```bash
python -m unittest
```
