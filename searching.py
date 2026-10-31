def breadthFirstSearch(graph, start):
    visited = set()
    queue = [start]
    result = []

    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            result.append(vertex)
            queue.extend(neighbor for neighbor in graph[vertex] if neighbor not in visited)

    return result

def depthFirstSearch(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    result = [start]

    for neighbor in graph[start]:
        if neighbor not in visited:
            result.extend(depthFirstSearch(graph, neighbor, visited))

    return result

def treeSearch(graph, start):
    visited = set()
    stack = [start]
    result = []

    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            result.append(vertex)
            stack.extend(neighbor for neighbor in graph[vertex] if neighbor not in visited)

    return result

def graphSearch(graph, start):
    visited = set()
    stack = [start]
    result = []

    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            result.append(vertex)
            stack.extend(neighbor for neighbor in graph[vertex] if neighbor not in visited)

    return result

def uniformCostSearch(graph, start):
    import heapq

    visited = set()
    priority_queue = [(0, start)]  # (cost, vertex)
    result = []

    while priority_queue:
        cost, vertex = heapq.heappop(priority_queue)
        if vertex not in visited:
            visited.add(vertex)
            result.append((vertex, cost))
            for neighbor, edge_cost in graph[vertex].items():
                if neighbor not in visited:
                    heapq.heappush(priority_queue, (cost + edge_cost, neighbor))

    return result

def depthLimitedSearch(graph, start, limit, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    result = [start]

    if limit <= 0:
        return result

    for neighbor in graph[start]:
        if neighbor not in visited:
            result.extend(depthLimitedSearch(graph, neighbor, limit - 1, visited))

    return result

def iterativeDeepeningSearch(graph, start, max_depth):
    result = []
    for depth in range(max_depth + 1):
        result.extend(depthLimitedSearch(graph, start, depth))
    return result

def iterativeDeepeningAStarSearch(graph, start, heuristic, max_depth):
    import heapq

    def a_star_search(graph, start, heuristic, depth_limit):
        visited = set()
        priority_queue = [(heuristic[start], 0, start)]  # (f_score, g_score, vertex)
        result = []

        while priority_queue:
            f_score, g_score, vertex = heapq.heappop(priority_queue)
            if vertex not in visited:
                visited.add(vertex)
                result.append((vertex, g_score))
                if g_score < depth_limit:
                    for neighbor, edge_cost in graph[vertex].items():
                        if neighbor not in visited:
                            new_g_score = g_score + edge_cost
                            new_f_score = new_g_score + heuristic[neighbor]
                            heapq.heappush(priority_queue, (new_f_score, new_g_score, neighbor))

        return result

    final_result = []
    for depth in range(max_depth + 1):
        final_result.extend(a_star_search(graph, start, heuristic, depth))
    return final_result


def iterativeDeepeningAStarSearch(graph, start, heuristic, max_depth):
    import heapq

    def a_star_search(graph, start, heuristic, depth_limit):
        visited = set()
        priority_queue = [(heuristic[start], 0, start)]  # (f_score, g_score, vertex)
        result = []

        while priority_queue:
            f_score, g_score, vertex = heapq.heappop(priority_queue)
            if vertex not in visited:
                visited.add(vertex)
                result.append((vertex, g_score))
                if g_score < depth_limit:
                    for neighbor, edge_cost in graph[vertex].items():
                        if neighbor not in visited:
                            new_g_score = g_score + edge_cost
                            new_f_score = new_g_score + heuristic[neighbor]
                            heapq.heappush(priority_queue, (new_f_score, new_g_score, neighbor))

        return result

    final_result = []
    for depth in range(max_depth + 1):
        final_result.extend(a_star_search(graph, start, heuristic, depth))
    return final_result

def aStarSearch(graph, start, heuristic):
    import heapq

    visited = set()
    priority_queue = [(heuristic[start], 0, start)]  # (f_score, g_score, vertex)
    result = []

    while priority_queue:
        f_score, g_score, vertex = heapq.heappop(priority_queue)
        if vertex not in visited:
            visited.add(vertex)
            result.append((vertex, g_score))
            for neighbor, edge_cost in graph[vertex].items():
                if neighbor not in visited:
                    new_g_score = g_score + edge_cost
                    new_f_score = new_g_score + heuristic[neighbor]
                    heapq.heappush(priority_queue, (new_f_score, new_g_score, neighbor))

    return result


