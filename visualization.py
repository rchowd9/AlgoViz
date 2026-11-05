import heapq
import json
import math
from itertools import count

from searching import binarySearch, fibonacciSearch, linearSearch, ternarySearch
from sorting import bucket_sort, counting_sort, heap_sort, pigeonhole_sort, radixSort, shell_sort


SEARCH_ALGORITHMS = (
    "Breadth-first search",
    "Depth-first search",
    "Uniform-cost search",
    "Greedy best-first search",
    "A* search",
    "Depth-limited search",
    "Iterative deepening search",
    "Iterative deepening A* search",
    "Bidirectional search",
)
ARRAY_SEARCH_ALGORITHMS = (
    "Linear search",
    "Binary search",
    "Ternary search",
    "Fibonacci search",
)


def parse_graph(graph_data):
    if isinstance(graph_data, str):
        try:
            graph_data = json.loads(graph_data)
        except json.JSONDecodeError as error:
            raise ValueError(f"Graph JSON is invalid: {error.msg}.") from error

    if not isinstance(graph_data, dict) or not graph_data:
        raise ValueError("The graph must be a non-empty JSON object.")

    graph = {}
    for node, neighbors in graph_data.items():
        if not isinstance(node, str) or not node or not isinstance(neighbors, dict):
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


def _validate_search(graph, start, goal, algorithm, heuristic, max_depth):
    if not isinstance(start, str) or not isinstance(goal, str):
        raise ValueError("The starting node and goal must be node names.")
    if start not in graph or goal not in graph:
        raise ValueError("The starting node and goal must both exist in the graph.")
    if algorithm not in SEARCH_ALGORITHMS:
        raise ValueError(f"Unknown search algorithm: {algorithm}.")
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0:
        raise ValueError("Maximum depth must be a non-negative integer.")
    if algorithm in {"Greedy best-first search", "A* search", "Iterative deepening A* search"}:
        if not isinstance(heuristic, dict):
            raise ValueError("Provide a heuristic value for every graph node.")
        for node in graph:
            value = heuristic.get(node)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("Heuristics must be finite, non-negative numbers for every node.")


def _result(order, path, costs, trace=None):
    return {
        "order": order,
        "path": path,
        "cost": costs.get(path[-1]) if path else None,
        "found": path is not None,
        "expanded": len(order),
        "trace": trace or [],
    }


def search_graph(graph, start, goal, algorithm, heuristic=None, max_depth=20):
    graph = parse_graph(graph)
    _validate_search(graph, start, goal, algorithm, heuristic, max_depth)

    if algorithm == "Bidirectional search":
        return _bidirectional_search(graph, start, goal)
    if algorithm in {"Depth-limited search", "Iterative deepening search"}:
        return _limited_search(graph, start, goal, algorithm, max_depth)
    if algorithm == "Iterative deepening A* search":
        return _ida_star(graph, start, goal, heuristic, max_depth)
    if algorithm in {
        "Greedy best-first search",
        "A* search",
        "Uniform-cost search",
    }:
        return _priority_search(graph, start, goal, algorithm, heuristic)
    return _uninformed_search(graph, start, goal, algorithm)


def search_array(values, target, algorithm):
    if algorithm not in ARRAY_SEARCH_ALGORITHMS:
        raise ValueError(f"Unknown array search algorithm: {algorithm}.")
    if not values:
        raise ValueError("Enter at least one number to search.")
    if algorithm != "Linear search" and values != sorted(values):
        raise ValueError(f"{algorithm} requires the input sequence to be sorted.")

    search_functions = {
        "Linear search": linearSearch,
        "Binary search": binarySearch,
        "Ternary search": ternarySearch,
        "Fibonacci search": fibonacciSearch,
    }
    return search_functions[algorithm](values, target)


def _uninformed_search(graph, start, goal, algorithm):
    frontier = [start]
    parent = {start: None}
    costs = {start: 0}
    visited = set()
    order = []

    while frontier:
        node = frontier.pop() if algorithm == "Depth-first search" else frontier.pop(0)
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        if node == goal:
            break
        neighbors = graph[node]
        if algorithm == "Depth-first search":
            neighbors = reversed(list(neighbors))
        for neighbor in neighbors:
            if neighbor not in visited and neighbor not in parent:
                parent[neighbor] = node
                costs[neighbor] = costs[node] + graph[node][neighbor]
                frontier.append(neighbor)

    path = _reconstruct_path(parent, goal) if goal in visited else None
    return _result(order, path, costs)


def _priority_search(graph, start, goal, algorithm, heuristic):
    sequence = count()
    frontier = [(heuristic[start] if algorithm == "Greedy best-first search" else 0, next(sequence), start)]
    parent = {start: None}
    costs = {start: 0}
    visited = set()
    order = []

    while frontier:
        _, _, node = heapq.heappop(frontier)
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        if node == goal:
            break
        for neighbor, edge_cost in graph[node].items():
            new_cost = costs[node] + edge_cost
            if neighbor in visited or new_cost >= costs.get(neighbor, math.inf):
                continue
            costs[neighbor] = new_cost
            parent[neighbor] = node
            priority = (
                heuristic[neighbor]
                if algorithm == "Greedy best-first search"
                else new_cost + (heuristic[neighbor] if algorithm == "A* search" else 0)
            )
            heapq.heappush(frontier, (priority, next(sequence), neighbor))

    path = _reconstruct_path(parent, goal) if goal in visited else None
    return _result(order, path, costs)


def _limited_search(graph, start, goal, algorithm, max_depth):
    order = []
    costs = {}
    found_path = None
    depth_limits = range(max_depth + 1) if algorithm == "Iterative deepening search" else (max_depth,)

    for limit in depth_limits:
        stack = [(start, [start], 0)]
        while stack:
            node, path, cost = stack.pop()
            order.append(node)
            costs[node] = min(cost, costs.get(node, math.inf))
            if node == goal:
                found_path = path
                break
            if len(path) - 1 >= limit:
                continue
            for neighbor, edge_cost in reversed(list(graph[node].items())):
                if neighbor not in path:
                    stack.append((neighbor, path + [neighbor], cost + edge_cost))
        if found_path is not None:
            break

    return _result(order, found_path, costs)


def _ida_star(graph, start, goal, heuristic, max_depth):
    threshold = heuristic[start]
    order = []
    costs = {start: 0}
    found_path = None

    while True:
        next_threshold = math.inf

        def visit(node, path, cost):
            nonlocal next_threshold, found_path
            score = cost + heuristic[node]
            if score > threshold:
                next_threshold = min(next_threshold, score)
                return
            order.append(node)
            costs[node] = min(cost, costs.get(node, math.inf))
            if node == goal:
                found_path = path
                return
            if len(path) - 1 >= max_depth:
                return
            for neighbor, edge_cost in graph[node].items():
                if neighbor not in path and found_path is None:
                    visit(neighbor, path + [neighbor], cost + edge_cost)

        visit(start, [start], 0)
        if found_path is not None or next_threshold == math.inf:
            break
        threshold = next_threshold

    return _result(order, found_path, costs)


def _bidirectional_search(graph, start, goal):
    if start == goal:
        return _result([start], [start], {start: 0})

    reverse_graph = {node: [] for node in graph}
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            reverse_graph[neighbor].append(node)

    from_start = {start: None}
    from_goal = {goal: None}
    queue_start = [start]
    queue_goal = [goal]
    order = []
    meeting = None

    while queue_start and queue_goal and meeting is None:
        meeting = _expand_bidirectional_layer(
            graph, queue_start, from_start, from_goal, order
        )
        if meeting is None:
            meeting = _expand_bidirectional_layer(
                reverse_graph, queue_goal, from_goal, from_start, order
            )

    if meeting is None:
        return _result(order, None, {})
    left = _reconstruct_path(from_start, meeting)
    right = []
    node = from_goal[meeting]
    while node is not None:
        right.append(node)
        node = from_goal[node]
    path = left + right
    route_cost = sum(graph[node][neighbor] for node, neighbor in zip(path, path[1:]))
    return _result(order, path, {path[-1]: route_cost})


def _expand_bidirectional_layer(graph, queue, parents, other_parents, order):
    layer_size = len(queue)
    for _ in range(layer_size):
        node = queue.pop(0)
        order.append(node)
        if node in other_parents:
            return node
        for neighbor in graph[node]:
            if neighbor not in parents:
                parents[neighbor] = node
                queue.append(neighbor)
                if neighbor in other_parents:
                    return neighbor
    return None


def _reconstruct_path(parent, goal):
    if goal not in parent:
        return None
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return list(reversed(path))


def sorting_trace(values, algorithm):
    supported_algorithms = {
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
    }
    if algorithm not in supported_algorithms:
        raise ValueError(f"Unknown sorting algorithm: {algorithm}.")
    if not values:
        raise ValueError("Enter at least one number to sort.")

    items = list(values)
    steps = [{"values": items.copy(), "description": "Initial sequence"}]

    def record(description):
        steps.append({"values": items.copy(), "description": description})

    if algorithm == "Bubble sort":
        for end in range(len(items) - 1, 0, -1):
            changed = False
            for index in range(end):
                if items[index] > items[index + 1]:
                    items[index], items[index + 1] = items[index + 1], items[index]
                    changed = True
                    record(f"Swap positions {index} and {index + 1}")
            if not changed:
                break
    elif algorithm == "Insertion sort":
        for index in range(1, len(items)):
            value = items[index]
            position = index
            while position > 0 and items[position - 1] > value:
                items[position] = items[position - 1]
                position -= 1
            items[position] = value
            record(f"Insert {value} into the sorted prefix")
    elif algorithm == "Selection sort":
        for index in range(len(items) - 1):
            smallest = min(range(index, len(items)), key=items.__getitem__)
            items[index], items[smallest] = items[smallest], items[index]
            record(f"Place the minimum value at position {index}")
    elif algorithm == "Merge sort":
        def merge_sort(left, right):
            if right - left < 2:
                return
            middle = (left + right) // 2
            merge_sort(left, middle)
            merge_sort(middle, right)
            merged = sorted(items[left:right])
            items[left:right] = merged
            record(f"Merge positions {left} through {right - 1}")

        merge_sort(0, len(items))
    else:
        sort_functions = {
            "Shell sort": shell_sort,
            "Heap sort": heap_sort,
            "Counting sort": counting_sort,
            "Radix sort": radixSort,
            "Pigeonhole sort": pigeonhole_sort,
            "Bucket sort": bucket_sort,
        }
        sort_input = items.copy()
        sorted_items = sort_functions[algorithm](sort_input)
        if sorted_items is None:
            sorted_items = sort_input
        items[:] = sorted_items
        record(f"Complete {algorithm.lower()}")

    return steps


def hill_climb_trace(start, minimum=-10, maximum=10):
    if isinstance(start, bool) or not isinstance(start, int) or not minimum <= start <= maximum:
        raise ValueError(f"Starting position must be an integer from {minimum} to {maximum}.")

    points = [start]
    while True:
        current = points[-1]
        neighbors = [point for point in (current - 1, current + 1) if minimum <= point <= maximum]
        next_point = max(neighbors, key=lambda point: -(point - 3) ** 2 + 9, default=current)
        if -(next_point - 3) ** 2 + 9 <= -(current - 3) ** 2 + 9:
            return points
        points.append(next_point)
