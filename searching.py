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

def greedyBestFirstSearch(graph, start, heuristic):
    import heapq

    visited = set()
    priority_queue = [(heuristic[start], start)]  # (heuristic, vertex)
    result = []

    while priority_queue:
        heuristic_value, vertex = heapq.heappop(priority_queue)
        if vertex not in visited:
            visited.add(vertex)
            result.append(vertex)
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    heapq.heappush(priority_queue, (heuristic[neighbor], neighbor))

    return result

def reconstruct_path(parents_start, parents_goal, meeting_vertex):
    path_start = []
    current = meeting_vertex
    while current is not None:
        path_start.append(current)
        current = parents_start[current]
    path_start.reverse()

    current = parents_goal[meeting_vertex]
    while current is not None:
        path_start.append(current)
        current = parents_goal[current]

    return path_start


def bidirectionalSearch(graph, start, goal):
    from collections import deque

    if start == goal:
        return [start]

    visited_start = {start}
    visited_goal = {goal}
    queue_start = deque([start])
    queue_goal = deque([goal])
    parents_start = {start: None}
    parents_goal = {goal: None}

    while queue_start and queue_goal:
        # Expand from the start side
        current_start = queue_start.popleft()
        for neighbor in graph[current_start]:
            if neighbor not in visited_start:
                visited_start.add(neighbor)
                parents_start[neighbor] = current_start
                queue_start.append(neighbor)
                if neighbor in visited_goal:
                    return reconstruct_path(parents_start, parents_goal, neighbor)

        # Expand from the goal side
        current_goal = queue_goal.popleft()
        for neighbor in graph[current_goal]:
            if neighbor not in visited_goal:
                visited_goal.add(neighbor)
                parents_goal[neighbor] = current_goal
                queue_goal.append(neighbor)
                if neighbor in visited_start:
                    return reconstruct_path(parents_start, parents_goal, neighbor)

    return None  # No path found

def andOrGraphSearch(graph, start, goal):
    visited = set()
    stack = [(start, [start])]
    result = []

    while stack:
        vertex, path = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            if vertex == goal:
                result.append(path)
            else:
                for neighbor in graph[vertex]:
                    if neighbor not in visited:
                        stack.append((neighbor, path + [neighbor]))

    return result

def linearSearch(arr, targetVal):
    for i in range(len(arr)):
        if arr[i] == targetVal:
            return i
    return -1

def binarySearch(arr, targetVal):
    left = 0
    right = len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == targetVal:
            return mid
        
        if arr[mid] < targetVal:
            left = mid + 1
        else:
            right = mid - 1

    return -1

def ternarySearch(arr, targetVal):
    left = 0
    right = len(arr) - 1

    while left <= right:
        third = (right - left) // 3
        mid1 = left + third
        mid2 = right - third

        if arr[mid1] == targetVal:
            return mid1
        if arr[mid2] == targetVal:
            return mid2

        if targetVal < arr[mid1]:
            right = mid1 - 1
        elif targetVal > arr[mid2]:
            left = mid2 + 1
        else:
            left = mid1 + 1
            right = mid2 - 1

    return -1

def fibonacciSearch(arr, targetVal):
    fibM2 = 0  # (m-2)'th Fibonacci number
    fibM1 = 1  # (m-1)'th Fibonacci number
    fibM = fibM2 + fibM1  # m'th Fibonacci number

    while fibM < len(arr):
        fibM2 = fibM1
        fibM1 = fibM
        fibM = fibM2 + fibM1

    offset = -1

    while fibM > 1:
        i = min(offset + fibM2, len(arr) - 1)

        if arr[i] < targetVal:
            fibM = fibM1
            fibM1 = fibM2
            fibM2 = fibM - fibM1
            offset = i
        elif arr[i] > targetVal:
            fibM = fibM2
            fibM1 -= 1
            fibM2 -= 1
        else:
            return i

    if fibM1 and offset + 1 < len(arr) and arr[offset + 1] == targetVal:
        return offset + 1

    return -1

import heapq

def b_star(graph, start, goal, heuristic):
    """
    B* (Best-First Graph Search) algorithm.
    graph: adjacency list {node: [(neighbor, cost), ...]}
    heuristic: function(node) -> estimated cost to goal
    """
    frontier = []
    heapq.heappush(frontier, (heuristic[start], start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        f, current = heapq.heappop(frontier)

        if current == goal:
            break

        for neighbor, cost in graph.get(current, []):
            new_cost = cost_so_far[current] + cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                f_neighbor = new_cost + heuristic[neighbor]
                heapq.heappush(frontier, (f_neighbor, neighbor))
                came_from[neighbor] = current

    # Reconstruct path
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came_from.get(node)
    path.reverse()

    return path, cost_so_far.get(goal, float('inf'))

class AVLtree:
    """A self-balancing binary search tree (AVL), storing unique keys."""

    class Node:
        def __init__(self, key):
            self.key = key
            self.left = None
            self.right = None
            self.height = 1

    def __init__(self):
        self.root = None

    @staticmethod
    def _height(node):
        return node.height if node else 0

    def _update(self, node):
        node.height = 1 + max(self._height(node.left), self._height(node.right))

    def _rotate_right(self, root):
        pivot = root.left
        root.left = pivot.right
        pivot.right = root
        self._update(root)
        self._update(pivot)
        return pivot

    def _rotate_left(self, root):
        pivot = root.right
        root.right = pivot.left
        pivot.left = root
        self._update(root)
        self._update(pivot)
        return pivot

    def _balance(self, node):
        self._update(node)
        balance = self._height(node.left) - self._height(node.right)
        if balance > 1:
            if self._height(node.left.left) < self._height(node.left.right):
                node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        if balance < -1:
            if self._height(node.right.right) < self._height(node.right.left):
                node.right = self._rotate_right(node.right)
            return self._rotate_left(node)
        return node

    def _insert(self, node, key):
        if node is None:
            return self.Node(key)
        if key < node.key:
            node.left = self._insert(node.left, key)
        elif key > node.key:
            node.right = self._insert(node.right, key)
        else:
            return node
        return self._balance(node)

    def insert(self, key):
        """Insert a key; duplicate keys are ignored. Returns the root node."""
        self.root = self._insert(self.root, key)
        return self.root

    def _delete(self, node, key):
        if node is None:
            return None
        if key < node.key:
            node.left = self._delete(node.left, key)
        elif key > node.key:
            node.right = self._delete(node.right, key)
        else:
            if node.left is None:
                return node.right
            if node.right is None:
                return node.left
            successor = node.right
            while successor.left:
                successor = successor.left
            node.key = successor.key
            node.right = self._delete(node.right, successor.key)
        return self._balance(node)

    def delete(self, key):
        """Remove a key if present; returns the root node."""
        self.root = self._delete(self.root, key)
        return self.root

    def search(self, key):
        node = self.root
        while node:
            if key == node.key:
                return node
            node = node.left if key < node.key else node.right
        return None

    def inorder(self):
        """Return the keys in ascending order."""
        result = []

        def traverse(node):
            if node:
                traverse(node.left)
                result.append(node.key)
                traverse(node.right)

        traverse(self.root)
        return result

    def __contains__(self, key):
        return self.search(key) is not None


class RedBlackTree:
    """A red-black binary search tree, storing unique keys."""

    class Node:
        def __init__(self, key=None, color="BLACK"):
            self.key = key
            self.color = color
            self.left = self.right = self.parent = None

    def __init__(self):
        self.NIL = self.Node()
        self.root = self.NIL

    def _left_rotate(self, node):
        pivot = node.right
        node.right = pivot.left
        if pivot.left is not self.NIL:
            pivot.left.parent = node
        pivot.parent = node.parent
        if node.parent is self.NIL:
            self.root = pivot
        elif node is node.parent.left:
            node.parent.left = pivot
        else:
            node.parent.right = pivot
        pivot.left = node
        node.parent = pivot

    def _right_rotate(self, node):
        pivot = node.left
        node.left = pivot.right
        if pivot.right is not self.NIL:
            pivot.right.parent = node
        pivot.parent = node.parent
        if node.parent is self.NIL:
            self.root = pivot
        elif node is node.parent.right:
            node.parent.right = pivot
        else:
            node.parent.left = pivot
        pivot.right = node
        node.parent = pivot

    def search(self, key):
        node = self.root
        while node is not self.NIL:
            if key == node.key:
                return node
            node = node.left if key < node.key else node.right
        return None

    def insert(self, key):
        """Insert a key; duplicate keys are ignored. Returns its node."""
        parent = self.NIL
        node = self.root
        while node is not self.NIL:
            parent = node
            if key == node.key:
                return node
            node = node.left if key < node.key else node.right

        inserted = self.Node(key, "RED")
        inserted.left = inserted.right = self.NIL
        inserted.parent = parent
        if parent is self.NIL:
            self.root = inserted
        elif key < parent.key:
            parent.left = inserted
        else:
            parent.right = inserted
        self._insert_fixup(inserted)
        return inserted

    def _insert_fixup(self, node):
        while node.parent.color == "RED":
            if node.parent is node.parent.parent.left:
                uncle = node.parent.parent.right
                if uncle.color == "RED":
                    node.parent.color = uncle.color = "BLACK"
                    node.parent.parent.color = "RED"
                    node = node.parent.parent
                else:
                    if node is node.parent.right:
                        node = node.parent
                        self._left_rotate(node)
                    node.parent.color = "BLACK"
                    node.parent.parent.color = "RED"
                    self._right_rotate(node.parent.parent)
            else:
                uncle = node.parent.parent.left
                if uncle.color == "RED":
                    node.parent.color = uncle.color = "BLACK"
                    node.parent.parent.color = "RED"
                    node = node.parent.parent
                else:
                    if node is node.parent.left:
                        node = node.parent
                        self._right_rotate(node)
                    node.parent.color = "BLACK"
                    node.parent.parent.color = "RED"
                    self._left_rotate(node.parent.parent)
        self.root.color = "BLACK"
        self.root.parent = self.NIL

    def _minimum(self, node):
        while node.left is not self.NIL:
            node = node.left
        return node

    def _transplant(self, old, new):
        if old.parent is self.NIL:
            self.root = new
        elif old is old.parent.left:
            old.parent.left = new
        else:
            old.parent.right = new
        new.parent = old.parent

    def delete(self, key):
        """Remove a key if present; returns whether it was found."""
        target = self.search(key)
        if target is None:
            return False
        moved = target
        original_color = moved.color
        if target.left is self.NIL:
            fix = target.right
            self._transplant(target, target.right)
        elif target.right is self.NIL:
            fix = target.left
            self._transplant(target, target.left)
        else:
            moved = self._minimum(target.right)
            original_color = moved.color
            fix = moved.right
            if moved.parent is target:
                fix.parent = moved
            else:
                self._transplant(moved, moved.right)
                moved.right = target.right
                moved.right.parent = moved
            self._transplant(target, moved)
            moved.left = target.left
            moved.left.parent = moved
            moved.color = target.color
        if original_color == "BLACK":
            self._delete_fixup(fix)
        return True

    def _delete_fixup(self, node):
        while node is not self.root and node.color == "BLACK":
            if node is node.parent.left:
                sibling = node.parent.right
                if sibling.color == "RED":
                    sibling.color = "BLACK"
                    node.parent.color = "RED"
                    self._left_rotate(node.parent)
                    sibling = node.parent.right
                if sibling.left.color == "BLACK" and sibling.right.color == "BLACK":
                    sibling.color = "RED"
                    node = node.parent
                else:
                    if sibling.right.color == "BLACK":
                        sibling.left.color = "BLACK"
                        sibling.color = "RED"
                        self._right_rotate(sibling)
                        sibling = node.parent.right
                    sibling.color = node.parent.color
                    node.parent.color = "BLACK"
                    sibling.right.color = "BLACK"
                    self._left_rotate(node.parent)
                    node = self.root
            else:
                sibling = node.parent.left
                if sibling.color == "RED":
                    sibling.color = "BLACK"
                    node.parent.color = "RED"
                    self._right_rotate(node.parent)
                    sibling = node.parent.left
                if sibling.right.color == "BLACK" and sibling.left.color == "BLACK":
                    sibling.color = "RED"
                    node = node.parent
                else:
                    if sibling.left.color == "BLACK":
                        sibling.right.color = "BLACK"
                        sibling.color = "RED"
                        self._left_rotate(sibling)
                        sibling = node.parent.left
                    sibling.color = node.parent.color
                    node.parent.color = "BLACK"
                    sibling.left.color = "BLACK"
                    self._right_rotate(node.parent)
                    node = self.root
        node.color = "BLACK"

    def inorder(self):
        """Return the keys in ascending order."""
        result = []

        def traverse(node):
            if node is not self.NIL:
                traverse(node.left)
                result.append(node.key)
                traverse(node.right)

        traverse(self.root)
        return result

    def __contains__(self, key):
        return self.search(key) is not None


AVLTree = AVLtree