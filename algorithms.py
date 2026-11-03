import random
import math
import random


def geneticAlgorithm(population, fitness_function, generations, mutation_rate):
    """
    Implements a simple genetic algorithm.

    Parameters:
    - population: List of individuals (solutions).
    - fitness_function: Function to evaluate the fitness of an individual.
    - generations: Number of generations to run the algorithm.
    - mutation_rate: Probability of mutation for each individual.

    Returns:
    - The best individual found after the specified number of generations.
    """
    def crossover(parent1, parent2):
        """Create a child by combining two sequence-based individuals."""
        crossover_point = random.randint(0, min(len(parent1), len(parent2)))
        return parent1[:crossover_point] + parent2[crossover_point:]

    def mutate(individual):
        """Swap two positions in a sequence-based individual."""
        if len(individual) < 2:
            return individual

        first, second = random.sample(range(len(individual)), 2)
        mutated = list(individual)
        mutated[first], mutated[second] = mutated[second], mutated[first]

        if isinstance(individual, str):
            return "".join(mutated)
        if isinstance(individual, tuple):
            return tuple(mutated)
        return mutated

    for generation in range(generations):
        # Evaluate fitness of the population
        fitness_scores = [fitness_function(individual) for individual in population]
        
        # Select parents based on fitness
        def select_parents(population, fitness_scores):
            ...

        parents = select_parents(population, fitness_scores)
        
        # Create next generation through crossover
        next_generation = []
        while len(next_generation) < len(population):
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            next_generation.append(child)
        
        # Apply mutation
        for i in range(len(next_generation)):
            if random.random() < mutation_rate:
                next_generation[i] = mutate(next_generation[i])
        
        population = next_generation
    
    # Return the best individual from the final population
    best_individual = max(population, key=fitness_function)
    return best_individual

def miniMaxAlgorithm(state, depth, maximizing_player, evaluate_function):
    """
    Implements the minimax algorithm for decision making in two-player games.

    Parameters:
    - state: The current state of the game.
    - depth: The maximum depth to search.
    - maximizing_player: Boolean indicating if the current player is maximizing or minimizing.
    - evaluate_function: Function to evaluate the score of a given state.

    Returns:
    - The best score for the current player at the given state.
    """
    if depth == 0:
        return evaluate_function(state)

    children = state.get_children()
    if not children:
        return evaluate_function(state)

    if maximizing_player:
        max_eval = float('-inf')
        for child in children:
            eval = miniMaxAlgorithm(child, depth - 1, False, evaluate_function)
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for child in children:
            eval = miniMaxAlgorithm(child, depth - 1, True, evaluate_function)
            min_eval = min(min_eval, eval)
        return min_eval


def alphaBetaPruning(state, depth, alpha, beta, maximizing_player, evaluate_function):
    """
    Implements the minimax algorithm with alpha-beta pruning for decision making in two-player games.

    Parameters:
    - state: The current state of the game.
    - depth: The maximum depth to search.
    - alpha: The best value that the maximizer currently can guarantee at that level or above.
    - beta: The best value that the minimizer currently can guarantee at that level or above.
    - maximizing_player: Boolean indicating if the current player is maximizing or minimizing.
    - evaluate_function: Function to evaluate the score of a given state.

    Returns:
    - The best score for the current player at the given state.
    """
    if depth == 0:
        return evaluate_function(state)

    children = state.get_children()
    if not children:
        return evaluate_function(state)

    if maximizing_player:
        max_eval = float('-inf')
        for child in children:
            eval = alphaBetaPruning(child, depth - 1, alpha, beta, False, evaluate_function)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cut-off
        return max_eval
    else:
        min_eval = float('inf')
        for child in children:
            eval = alphaBetaPruning(child, depth - 1, alpha, beta, True, evaluate_function)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cut-off
        return min_eval


def hillClimbing(state, evaluate_function):
    """
    Implements the hill climbing algorithm for optimization problems.

    Parameters:
    - state: The initial state of the problem.
    - evaluate_function: Function to evaluate the score of a given state.

    Returns:
    - The best state found after hill climbing.
    """
    current_state = state
    current_score = evaluate_function(current_state)

    while True:
        neighbors = current_state.get_neighbors()
        if not neighbors:
            break

        next_state = max(neighbors, key=evaluate_function)
        next_score = evaluate_function(next_state)

        if next_score <= current_score:
            break  # No better neighbor found

        current_state = next_state
        current_score = next_score

    return current_state

def simulatedAnnealing(state, evaluate_function, initial_temperature, cooling_rate):
    """
    Implements the simulated annealing algorithm for optimization problems.

    Parameters:
    - state: The initial state of the problem.
    - evaluate_function: Function to evaluate the score of a given state.
    - initial_temperature: The starting temperature for the annealing process.
    - cooling_rate: The rate at which the temperature decreases.

    Returns:
    - The best state found after simulated annealing.
    """
    current_state = state
    current_score = evaluate_function(current_state)
    temperature = initial_temperature

    while temperature > 1:
        neighbors = current_state.get_neighbors()
        if not neighbors:
            break

        next_state = random.choice(neighbors)
        next_score = evaluate_function(next_state)

        # Calculate acceptance probability
        if next_score > current_score:
            acceptance_probability = 1.0
        else:
            acceptance_probability = math.exp((next_score - current_score) / temperature)

        # Decide whether to accept the new state
        if random.random() < acceptance_probability:
            current_state = next_state
            current_score = next_score

        # Cool down the temperature
        temperature *= cooling_rate

    return current_state

def dijikstra(graph, start):
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

def localBeamSearch(graph, start, beam_width, evaluate_function):
    """
    Implements the local beam search algorithm.

    Parameters:
    - graph: A dictionary representing the graph where keys are vertices and values are lists of neighbors.
    - start: The starting vertex for the search.
    - beam_width: The number of best states to keep at each level.
    - evaluate_function: Function that scores a state for selection.

    Returns:
    - A list of the best states found after the search.
    """
    current_states = [start]
    result = []

    while current_states:
        next_states = []
        for state in current_states:
            result.append(state)
            next_states.extend(graph[state])

        # Keep only the best 'beam_width' states
        current_states = sorted(next_states, key=lambda x: evaluate_function(x))[:beam_width]

    return result

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next          # 1 step
        fast = fast.next.next     # 2 steps
        if slow == fast:
            return True          # cycle found
    return False                 # no cycle

def h_minimax(state, depth, maximizing_player, evaluate_function, heuristic_function=None):
    """
    Implements the heuristic minimax algorithm for two-player zero-sum games.

    Parameters:
    - state: The current game state.
    - depth: The maximum depth to search.
    - maximizing_player: Boolean indicating whether the current turn is maximizing.
    - evaluate_function: Function used to evaluate a state when no heuristic is supplied.
    - heuristic_function: Optional function used to estimate the value of a state.

    Returns:
    - The best score for the current player from the given state.
    """
    if depth == 0:
        if heuristic_function is not None:
            return heuristic_function(state)
        return evaluate_function(state)

    children = state.get_children() if hasattr(state, "get_children") else []
    if not children:
        if heuristic_function is not None:
            return heuristic_function(state)
        return evaluate_function(state)

    if maximizing_player:
        best_score = float('-inf')
        for child in children:
            score = h_minimax(child, depth - 1, False, evaluate_function, heuristic_function)
            best_score = max(best_score, score)
        return best_score

    best_score = float('inf')
    for child in children:
        score = h_minimax(child, depth - 1, True, evaluate_function, heuristic_function)
        best_score = min(best_score, score)
    return best_score

def beliefStateUpdate(prior_belief, observation, transition_model, observation_model):
    """
    Updates the belief state based on the prior belief, observation, transition model, and observation model.

    Parameters:
    - prior_belief: A dictionary representing the prior belief state (state -> probability).
    - observation: The current observation.
    - transition_model: A function that takes a state and returns a dictionary of next states with their probabilities.
    - observation_model: A function that takes a state and an observation and returns the probability of that observation given the state.

    Returns:
    - A new belief state as a dictionary (state -> updated probability).
    """
    new_belief = {}
    total_probability = 0.0

    for state in prior_belief:
        # Predict step: calculate the predicted belief for each possible next state
        predicted_belief = 0.0
        for next_state, transition_prob in transition_model(state).items():
            predicted_belief += prior_belief[state] * transition_prob

        # Update step: incorporate the observation
        observation_prob = observation_model(state, observation)
        new_belief[state] = predicted_belief * observation_prob
        total_probability += new_belief[state]

    # Normalize the new belief state
    if total_probability > 0:
        for state in new_belief:
            new_belief[state] /= total_probability

    return new_belief

def kNearestNeighbors(data, query_point, k):
    """
    Implements the k-Nearest Neighbors algorithm.

    Parameters:
    - data: A list of tuples where each tuple contains a point and its label (point, label).
    - query_point: The point for which we want to find the nearest neighbors.
    - k: The number of nearest neighbors to find.

    Returns:
    - A list of the k nearest neighbors' labels.
    """
    from collections import Counter
    import math 

    

    def euclidean_distance(point1, point2): 
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

    # Calculate distances from the query point to all points in the dataset
    distances = [(euclidean_distance(point, query_point), label) for point, label in data]
    
    # Sort by distance and select the k nearest neighbors
    distances.sort(key=lambda x: x[0])
    k_nearest_neighbors = distances[:k]

    # Extract the labels of the k nearest neighbors
    labels = [label for _, label in k_nearest_neighbors]

    # Return the most common label among the k nearest neighbors
    most_common_label = Counter(labels).most_common(1)[0][0]
    
    return most_common_label

def euclideanAlgorithm(data, query_point):
    """
    Implements a simple nearest neighbor search using Euclidean distance.

    Parameters:
    - data: A list of tuples where each tuple contains a point and its label (point, label).
    - query_point: The point for which we want to find the nearest neighbor.

    Returns:
    - The label of the nearest neighbor.
    """
    import math

    def euclidean_distance(point1, point2):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

    # Find the nearest neighbor
    nearest_neighbor = min(data, key=lambda x: euclidean_distance(x[0], query_point))
    
    return nearest_neighbor[1]

def binaryGCDAlgorithm (a, b):
    """
    Implements the binary GCD algorithm (Stein's algorithm) to compute the greatest common divisor of two integers.

    Parameters:
    - a: First integer.
    - b: Second integer.

    Returns:
    - The greatest common divisor of a and b.
    """
    if a == 0:
        return b
    if b == 0:
        return a

    # Find the greatest power of 2 that divides both a and b
    shift = 0
    while ((a | b) & 1) == 0:
        a >>= 1
        b >>= 1
        shift += 1

    while (a & 1) == 0:
        a >>= 1

    while b != 0:
        while (b & 1) == 0:
            b >>= 1

        if a > b:
            a, b = b, a

        b -= a

    return a << shift

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