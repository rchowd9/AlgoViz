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

