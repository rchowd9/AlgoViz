import random
import math
import numbers
from collections import Counter

import numpy as np


def geneticAlgorithm(population, fitness_function, generations, mutation_rate, trace=None):
    """
    Implements a simple genetic algorithm.

    Parameters:
    - population: List of individuals (solutions).
    - fitness_function: Function to evaluate the fitness of an individual.
    - generations: Number of generations to run the algorithm.
    - mutation_rate: Probability of mutation for each individual.
    - trace: Optional list to receive the best and average fitness per generation.

    Returns:
    - The best individual found after the specified number of generations.
    """
    population = list(population)
    if not population:
        raise ValueError("The population must contain at least one individual.")
    if isinstance(generations, bool) or not isinstance(generations, int) or generations < 0:
        raise ValueError("Generations must be a non-negative integer.")
    if (
        isinstance(mutation_rate, bool)
        or not isinstance(mutation_rate, (int, float))
        or not math.isfinite(mutation_rate)
        or not 0 <= mutation_rate <= 1
    ):
        raise ValueError("Mutation rate must be a number between 0 and 1.")

    def crossover(parent1, parent2):
        """Create a child by combining two sequence-based individuals."""
        crossover_point = random.randint(0, min(len(parent1), len(parent2)))
        return parent1[:crossover_point] + parent2[crossover_point:]

    def mutate(individual):
        """Mutate a binary string by flipping one bit, or swap sequence positions."""
        if isinstance(individual, str) and set(individual) <= {"0", "1"}:
            if not individual:
                return individual
            position = random.randrange(len(individual))
            bit = "1" if individual[position] == "0" else "0"
            return individual[:position] + bit + individual[position + 1:]

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

    def record_generation(generation, individuals, fitness_scores):
        if trace is not None:
            best_index = max(range(len(individuals)), key=fitness_scores.__getitem__)
            trace.append(
                {
                    "generation": generation,
                    "best_individual": individuals[best_index],
                    "best_fitness": fitness_scores[best_index],
                    "average_fitness": sum(fitness_scores) / len(fitness_scores),
                }
            )

    fitness_scores = [fitness_function(individual) for individual in population]
    best_index = max(range(len(population)), key=fitness_scores.__getitem__)
    best_individual = population[best_index]
    best_fitness = fitness_scores[best_index]
    record_generation(0, population, fitness_scores)

    for generation in range(1, generations + 1):
        # Select parents with two-way tournaments, allowing a one-member population.
        def select_parent():
            candidates = random.sample(range(len(population)), min(2, len(population)))
            winner = max(candidates, key=fitness_scores.__getitem__)
            return population[winner]

        next_generation = [best_individual]
        while len(next_generation) < len(population):
            child = crossover(select_parent(), select_parent())
            if random.random() < mutation_rate:
                child = mutate(child)
            next_generation.append(child)

        population = next_generation
        fitness_scores = [fitness_function(individual) for individual in population]
        generation_best_index = max(range(len(population)), key=fitness_scores.__getitem__)
        if fitness_scores[generation_best_index] > best_fitness:
            best_individual = population[generation_best_index]
            best_fitness = fitness_scores[generation_best_index]
        record_generation(generation, population, fitness_scores)

    return best_individual

def miniMaxAlgorithm(state, depth, maximizing_player, evaluate_function, trace=None):
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
    def search(current_state, remaining_depth, maximizing, node_path):
        children = current_state.get_children() if remaining_depth != 0 else []
        if remaining_depth == 0 or not children:
            score = evaluate_function(current_state)
            if trace is not None:
                trace.append(
                    {
                        "node_path": node_path,
                        "depth": len(node_path),
                        "player": "Max" if maximizing else "Min",
                        "event": "Leaf evaluation",
                        "score": score,
                    }
                )
            return score

        child_scores = [
            search(child, remaining_depth - 1, not maximizing, node_path + (index,))
            for index, child in enumerate(children)
        ]
        score = max(child_scores) if maximizing else min(child_scores)
        if trace is not None:
            trace.append(
                {
                    "node_path": node_path,
                    "depth": len(node_path),
                    "player": "Max" if maximizing else "Min",
                    "event": "Select maximum" if maximizing else "Select minimum",
                    "score": score,
                }
            )
        return score

    return search(state, depth, maximizing_player, ())


def alphaBetaPruning(state, depth, alpha, beta, maximizing_player, evaluate_function, trace=None):
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
    def search(current_state, remaining_depth, alpha_value, beta_value, maximizing, node_path):
        if remaining_depth == 0:
            score = evaluate_function(current_state)
            if trace is not None:
                trace.append(
                    {
                        "node_path": node_path,
                        "depth": len(node_path),
                        "player": "Max" if maximizing else "Min",
                        "event": "Leaf evaluation",
                        "score": score,
                    }
                )
            return score

        children = current_state.get_children()
        if not children:
            score = evaluate_function(current_state)
            if trace is not None:
                trace.append(
                    {
                        "node_path": node_path,
                        "depth": len(node_path),
                        "player": "Max" if maximizing else "Min",
                        "event": "Leaf evaluation",
                        "score": score,
                    }
                )
            return score

        if maximizing:
            max_eval = float('-inf')
            for index, child in enumerate(children):
                eval_value = search(
                    child,
                    remaining_depth - 1,
                    alpha_value,
                    beta_value,
                    False,
                    node_path + (index,),
                )
                max_eval = max(max_eval, eval_value)
                alpha_value = max(alpha_value, eval_value)
                if beta_value <= alpha_value:
                    pruned = len(children) - index - 1
                    if trace is not None:
                        trace.append(
                            {
                                "node_path": node_path,
                                "depth": len(node_path),
                                "player": "Max",
                                "event": "Beta cut-off",
                                "score": max_eval,
                                "pruned": pruned,
                            }
                        )
                    break
            if trace is not None:
                trace.append(
                    {
                        "node_path": node_path,
                        "depth": len(node_path),
                        "player": "Max",
                        "event": "Select maximum",
                        "score": max_eval,
                    }
                )
            return max_eval

        min_eval = float('inf')
        for index, child in enumerate(children):
            eval_value = search(
                child,
                remaining_depth - 1,
                alpha_value,
                beta_value,
                True,
                node_path + (index,),
            )
            min_eval = min(min_eval, eval_value)
            beta_value = min(beta_value, eval_value)
            if beta_value <= alpha_value:
                pruned = len(children) - index - 1
                if trace is not None:
                    trace.append(
                        {
                            "node_path": node_path,
                            "depth": len(node_path),
                            "player": "Min",
                            "event": "Alpha cut-off",
                            "score": min_eval,
                            "pruned": pruned,
                        }
                    )
                break
        if trace is not None:
            trace.append(
                {
                    "node_path": node_path,
                    "depth": len(node_path),
                    "player": "Min",
                    "event": "Select minimum",
                    "score": min_eval,
                }
            )
        return min_eval

    return search(state, depth, alpha, beta, maximizing_player, ())


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

def simulatedAnnealing(state, evaluate_function, initial_temperature, cooling_rate, trace=None):
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
    if trace is not None:
        trace.append(
            {
                "temperature": temperature,
                "current_state": current_state,
                "current_score": current_score,
                "candidate_state": None,
                "candidate_score": None,
                "accepted": None,
                "result_state": current_state,
                "result_score": current_score,
            }
        )

    while temperature > 1:
        neighbors = current_state.get_neighbors()
        if not neighbors:
            break

        previous_state = current_state
        previous_score = current_score
        next_state = random.choice(neighbors)
        next_score = evaluate_function(next_state)

        # Calculate acceptance probability
        if next_score > current_score:
            acceptance_probability = 1.0
        else:
            acceptance_probability = math.exp((next_score - current_score) / temperature)

        # Decide whether to accept the new state
        accepted = random.random() < acceptance_probability
        if accepted:
            current_state = next_state
            current_score = next_score

        if trace is not None:
            trace.append(
                {
                    "temperature": temperature,
                    "current_state": previous_state,
                    "current_score": previous_score,
                    "candidate_state": next_state,
                    "candidate_score": next_score,
                    "accepted": accepted,
                    "result_state": current_state,
                    "result_score": current_score,
                }
            )

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
    predicted = {}
    for state, prior_probability in prior_belief.items():
        for next_state, transition_probability in transition_model(state).items():
            predicted[next_state] = predicted.get(next_state, 0.0) + (
                prior_probability * transition_probability
            )

    updated = {
        state: probability * observation_model(state, observation)
        for state, probability in predicted.items()
    }
    total_probability = sum(updated.values())
    if total_probability <= 0:
        raise ValueError("The observation has zero probability under the predicted belief.")
    return {state: probability / total_probability for state, probability in updated.items()}

def kNearestNeighbors(data, query_point, k, trace=None):
    """
    Implements the k-Nearest Neighbors algorithm.

    Parameters:
    - data: A list of tuples where each tuple contains a point and its label (point, label).
    - query_point: The point for which we want to find the nearest neighbors.
    - k: The number of nearest neighbors to find.

    Returns:
    - A list of the k nearest neighbors' labels.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")
    if not data:
        raise ValueError("Training data must not be empty.")
    if k > len(data):
        raise ValueError("k cannot exceed the number of training samples.")
    query = list(query_point)
    if not query:
        raise ValueError("Points must contain at least one coordinate.")

    distances = []
    for index, (point, label) in enumerate(data):
        point = list(point)
        if len(point) != len(query):
            raise ValueError("Training points and the query must have matching dimensions.")
        distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(point, query)))
        distances.append((distance, index, point, label))
    distances.sort(key=lambda item: (item[0], item[1]))
    k_nearest_neighbors = distances[:k]

    if trace is not None:
        trace.extend(
            {
                "rank": rank,
                "point": point,
                "label": label,
                "distance": distance,
            }
            for rank, (distance, _, point, label) in enumerate(distances, start=1)
        )

    # Extract the labels of the k nearest neighbors
    labels = [label for _, _, _, label in k_nearest_neighbors]
    return _majority_label(labels)

def euclideanAlgorithm(data, query_point):
    """
    Implements a simple nearest neighbor search using Euclidean distance.

    Parameters:
    - data: A list of tuples where each tuple contains a point and its label (point, label).
    - query_point: The point for which we want to find the nearest neighbor.

    Returns:
    - The label of the nearest neighbor.
    """
    if not data:
        raise ValueError("Training data must not be empty.")
    query = list(query_point)

    def distance(point):
        point = list(point)
        if len(point) != len(query):
            raise ValueError("Training points and the query must have matching dimensions.")
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(point, query)))

    nearest_neighbor = min(data, key=lambda item: distance(item[0]))
    return nearest_neighbor[1]

def binaryGCDAlgorithm(a, b):
    """
    Implements the binary GCD algorithm (Stein's algorithm) to compute the greatest common divisor of two integers.

    Parameters:
    - a: First integer.
    - b: Second integer.

    Returns:
    - The greatest common divisor of a and b.
    """
    if isinstance(a, bool) or not isinstance(a, int) or isinstance(b, bool) or not isinstance(b, int):
        raise ValueError("Both inputs must be integers.")
    a = abs(a)
    b = abs(b)
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

def huntsAlgorithm(graph, start):
    """
    Implements Hunt's algorithm for finding the shortest path in a graph.

    Parameters:
    - graph: A dictionary representing the graph where keys are vertices and values are lists of tuples (neighbor, weight).
    - start: The starting vertex for the search.

    Returns:
    - A dictionary mapping each vertex to its shortest distance from the start vertex.
    """
    import heapq

    distances = {vertex: float('inf') for vertex in graph}
    distances[start] = 0
    priority_queue = [(0, start)]  # (distance, vertex)

    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)

        if current_distance > distances[current_vertex]:
            continue

        for neighbor, weight in graph[current_vertex]:
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances

def bagging(data, labels, base_learner, n_estimators):
    """
    Implements the bagging ensemble method.

    Parameters:
    - data: A list of feature vectors (list of lists).
    - labels: A list of labels corresponding to the feature vectors.
    - base_learner: A function that takes data and labels and returns a trained model.
    - n_estimators: The number of base learners to train.

    Returns:
    - A list of trained base learners.
    """
    from random import choices

    models = []
    for _ in range(n_estimators):
        # Create a bootstrap sample
        indices = choices(range(len(data)), k=len(data))
        bootstrapped_data = [data[i] for i in indices]
        bootstrapped_labels = [labels[i] for i in indices]
        # Train the base learner on the bootstrap sample
        model = base_learner(bootstrapped_data, bootstrapped_labels)
        models.append(model)

    return models

class DecisionTree:
    """A small classification tree used by the RandomForest implementation."""

    def __init__(self, max_depth=None):
        if max_depth is not None and (
            isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0
        ):
            raise ValueError("Maximum depth must be a non-negative integer.")
        self.max_depth = max_depth
        self.tree = None
        self.classes_ = []

    def fit(self, X, y):
        rows, labels = _validated_training_data(X, y)
        self.classes_ = sorted(set(labels), key=str)

        def build(indices, depth):
            counts = {}
            for index in indices:
                counts[labels[index]] = counts.get(labels[index], 0) + 1
            majority = _majority_label([labels[index] for index in indices])
            impurity = 1 - sum((count / len(indices)) ** 2 for count in counts.values())
            node_data = {
                "samples": len(indices),
                "counts": counts,
                "impurity": impurity,
            }
            if len(counts) == 1 or (self.max_depth is not None and depth >= self.max_depth):
                return dict(node_data, value=majority)

            parent_impurity = impurity
            best_gain, best_split = 0, None
            for feature in range(len(rows[0])):
                try:
                    values = sorted({float(rows[index][feature]) for index in indices})
                except (TypeError, ValueError):
                    continue
                for left_value, right_value in zip(values, values[1:]):
                    threshold = (left_value + right_value) / 2
                    left = [i for i in indices if float(rows[i][feature]) <= threshold]
                    right = [i for i in indices if float(rows[i][feature]) > threshold]
                    if not left or not right:
                        continue
                    impurity = 0
                    for subset in (left, right):
                        subset_counts = {}
                        for index in subset:
                            subset_counts[labels[index]] = subset_counts.get(labels[index], 0) + 1
                        impurity += len(subset) / len(indices) * (
                            1 - sum((count / len(subset)) ** 2 for count in subset_counts.values())
                        )
                    gain = parent_impurity - impurity
                    if gain > best_gain:
                        best_gain, best_split = gain, (feature, threshold, left, right)

            if best_split is None:
                return dict(node_data, value=majority)
            feature, threshold, left, right = best_split
            return dict(node_data, **{
                "feature": feature,
                "threshold": threshold,
                "left": build(left, depth + 1),
                "right": build(right, depth + 1),
            })

        self.tree = build(list(range(len(rows))), 0)
        return self

    def predict(self, X):
        rows = X.tolist() if hasattr(X, "tolist") else list(X)
        if self.tree is None:
            raise ValueError("The decision tree must be fitted before prediction.")

        def classify(row):
            node = self.tree
            while "value" not in node:
                node = node["left"] if float(row[node["feature"]]) <= node["threshold"] else node["right"]
            return node["value"]

        return np.asarray([classify(row) for row in rows])

    def get_depth(self):
        if self.tree is None:
            raise ValueError("The decision tree must be fitted before inspecting it.")

        def depth(node):
            if "value" in node:
                return 0
            return 1 + max(depth(node["left"]), depth(node["right"]))

        return depth(self.tree)

    def get_n_leaves(self):
        if self.tree is None:
            raise ValueError("The decision tree must be fitted before inspecting it.")

        def count(node):
            if "value" in node:
                return 1
            return count(node["left"]) + count(node["right"])

        return count(self.tree)

    def score(self, X, y):
        predictions = self.predict(X).tolist()
        targets = y.tolist() if hasattr(y, "tolist") else list(y)
        if not targets or len(predictions) != len(targets):
            raise ValueError("X and y must contain the same non-zero number of samples.")
        return sum(prediction == target for prediction, target in zip(predictions, targets)) / len(targets)


def trainDecisionTreeClassifier(data, labels, max_depth=3):
    """Train a classification tree using Gini impurity and numeric splits."""
    return DecisionTree(max_depth=max_depth).fit(data, labels)


class RandomForest:
    def __init__(self, n_estimators=10, max_depth=5, bootstrap=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.trees = []

    def _bootstrap_sample(self, X, y):
        import numpy as np

        n = X.shape[0]
        indices = np.random.choice(n, size=n, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        rows, labels = _validated_training_data(X, y)
        if isinstance(self.n_estimators, bool) or not isinstance(self.n_estimators, int) or self.n_estimators < 1:
            raise ValueError("Number of estimators must be a positive integer.")
        self.trees = []
        for _ in range(self.n_estimators):
            if self.bootstrap:
                indices = [random.randrange(len(rows)) for _ in rows]
                X_boot = [rows[index] for index in indices]
                y_boot = [labels[index] for index in indices]
            else:
                X_boot, y_boot = rows, labels
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        self.classes_ = list(dict.fromkeys(labels))
        return self

    def predict(self, X):
        if not self.trees:
            raise ValueError("The random forest must be fitted before prediction.")
        predictions = [tree.predict(X).tolist() for tree in self.trees]
        return np.asarray(
            [
                _majority_label([tree_prediction[index] for tree_prediction in predictions])
                for index in range(len(predictions[0]))
            ]
        )

def iterativeDichotomiser3(data, labels, depth=0, max_depth=None):
    """
    Implements the Iterative Dichotomiser 3 (ID3) algorithm for decision tree learning.

    Parameters:
    - data: A list of feature vectors (list of lists).
    - labels: A list of labels corresponding to the feature vectors.
    - depth: The current depth of the tree (used for recursion).
    - max_depth: The maximum depth of the tree (optional).

    Returns:
    - A decision tree represented as a nested dictionary.
    """
    rows, targets = _validated_training_data(data, labels)
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
        raise ValueError("Depth must be a non-negative integer.")
    if max_depth is not None and (
        isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0
    ):
        raise ValueError("Maximum depth must be a non-negative integer.")

    def entropy(values):
        counts = Counter(values)
        total = len(values)
        return -sum((count / total) * math.log2(count / total) for count in counts.values())

    def build(subset, subset_labels, current_depth, available_features):
        majority = _majority_label(subset_labels)
        if (
            len(set(subset_labels)) == 1
            or not available_features
            or (max_depth is not None and current_depth >= max_depth)
        ):
            return majority

        parent_entropy = entropy(subset_labels)
        best_feature = None
        best_gain = 0.0
        best_groups = None
        for feature in available_features:
            groups = {}
            for row, label in zip(subset, subset_labels):
                groups.setdefault(row[feature], []).append((row, label))
            weighted_entropy = sum(
                len(group) / len(subset_labels) * entropy([label for _, label in group])
                for group in groups.values()
            )
            gain = parent_entropy - weighted_entropy
            if gain > best_gain:
                best_feature, best_gain, best_groups = feature, gain, groups

        if best_feature is None:
            return majority
        remaining = [feature for feature in available_features if feature != best_feature]
        branches = {
            value: build(
                [row for row, _ in group],
                [label for _, label in group],
                current_depth + 1,
                remaining,
            )
            for value, group in best_groups.items()
        }
        return {"feature": best_feature, "branches": branches, "default": majority}

    return build(rows, targets, depth, list(range(len(rows[0]))))

def CART(data, labels, depth=0, max_depth=None):
    """
    Implements the Classification and Regression Trees (CART) algorithm for decision tree learning.

    Parameters:
    - data: A list of feature vectors (list of lists).
    - labels: A list of labels corresponding to the feature vectors.
    - depth: The current depth of the tree (used for recursion).
    - max_depth: The maximum depth of the tree (optional).

    Returns:
    - A decision tree represented as a nested dictionary.
    """
    return _cart_classification_tree(data, labels, depth, max_depth)


def _validated_training_data(data, labels):
    rows = data.tolist() if hasattr(data, "tolist") else list(data)
    targets = labels.tolist() if hasattr(labels, "tolist") else list(labels)
    if not rows or not targets:
        raise ValueError("Training data and labels must not be empty.")
    if len(rows) != len(targets):
        raise ValueError("Training data and labels must have the same length.")
    if not rows[0]:
        raise ValueError("Training data must contain at least one feature.")
    feature_count = len(rows[0])
    if any(len(row) != feature_count for row in rows):
        raise ValueError("All training rows must have the same number of features.")
    try:
        for label in targets:
            hash(label)
    except TypeError as error:
        raise ValueError("Labels must be hashable.") from error
    return rows, targets


def _majority_label(labels):
    counts = Counter(labels)
    return max(counts, key=counts.get)


def _cart_classification_tree(data, labels, depth=0, max_depth=None):
    rows, targets = _validated_training_data(data, labels)
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
        raise ValueError("Depth must be a non-negative integer.")
    if max_depth is not None and (
        isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0
    ):
        raise ValueError("Maximum depth must be a non-negative integer.")

    def gini(values):
        counts = Counter(values)
        return 1 - sum((count / len(values)) ** 2 for count in counts.values())

    def build(indices, current_depth):
        current_labels = [targets[index] for index in indices]
        majority = _majority_label(current_labels)
        if (
            len(set(current_labels)) == 1
            or (max_depth is not None and current_depth >= max_depth)
        ):
            return majority

        parent_impurity = gini(current_labels)
        best_split = None
        best_gain = 0.0
        for feature in range(len(rows[0])):
            try:
                values = sorted({float(rows[index][feature]) for index in indices})
            except (TypeError, ValueError):
                continue
            for left_value, right_value in zip(values, values[1:]):
                threshold = (left_value + right_value) / 2
                left = [index for index in indices if float(rows[index][feature]) <= threshold]
                right = [index for index in indices if float(rows[index][feature]) > threshold]
                if not left or not right:
                    continue
                weighted = (
                    len(left) / len(indices) * gini([targets[index] for index in left])
                    + len(right) / len(indices) * gini([targets[index] for index in right])
                )
                gain = parent_impurity - weighted
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature, threshold, left, right)

        if best_split is None:
            return majority
        feature, threshold, left, right = best_split
        return {
            "feature": feature,
            "threshold": threshold,
            "left": build(left, current_depth + 1),
            "right": build(right, current_depth + 1),
            "default": majority,
        }

    return build(list(range(len(rows))), depth)

def gradientDescent(gradient_function, initial_point, learning_rate, iterations, trace=None):
    """
    Implements the gradient descent optimization algorithm.

    Parameters:
    - gradient_function: A function that computes the gradient at a given point.
    - initial_point: The starting point for the optimization.
    - learning_rate: The step size for each iteration.
    - iterations: The number of iterations to perform.
    - trace: Optional list to receive the initial point and each updated point.

    Returns:
    - The optimized point after performing gradient descent.
    """
    if not callable(gradient_function):
        raise ValueError("Gradient function must be callable.")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, numbers.Real)
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("Learning rate must be a positive finite number.")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
        raise ValueError("Iterations must be a non-negative integer.")

    scalar_point = isinstance(initial_point, numbers.Real) and not isinstance(initial_point, bool)
    if scalar_point:
        point = initial_point
        if not math.isfinite(point):
            raise ValueError("Initial point must contain finite numbers.")
        if trace is not None:
            trace.append(point)

        for _ in range(iterations):
            gradient = gradient_function(point)
            if isinstance(gradient, (list, tuple)) and len(gradient) == 1:
                gradient = gradient[0]
            if (
                isinstance(gradient, bool)
                or not isinstance(gradient, numbers.Real)
                or not math.isfinite(gradient)
            ):
                raise ValueError("Gradient function must return a finite number.")
            point -= learning_rate * gradient
            if not math.isfinite(point):
                raise ValueError("Gradient descent produced a non-finite point.")
            if trace is not None:
                trace.append(point)
        return point

    if not isinstance(initial_point, (list, tuple)):
        raise ValueError("Initial point must be a number or a list/tuple of numbers.")
    if not initial_point:
        raise ValueError("Initial point must contain at least one number.")

    point = list(initial_point)
    for coordinate in point:
        if (
            isinstance(coordinate, bool)
            or not isinstance(coordinate, numbers.Real)
            or not math.isfinite(coordinate)
        ):
            raise ValueError("Initial point must contain finite numbers.")
    if trace is not None:
        trace.append(point.copy())

    for _ in range(iterations):
        gradient = gradient_function(point)
        if not isinstance(gradient, (list, tuple)) or len(gradient) != len(point):
            raise ValueError("Gradient function must return one value per coordinate.")
        updated = []
        for coordinate, component in zip(point, gradient):
            if (
                isinstance(component, bool)
                or not isinstance(component, numbers.Real)
                or not math.isfinite(component)
            ):
                raise ValueError("Gradient function must return finite numbers.")
            next_coordinate = coordinate - learning_rate * component
            if not math.isfinite(next_coordinate):
                raise ValueError("Gradient descent produced a non-finite point.")
            updated.append(next_coordinate)
        point = updated
        if trace is not None:
            trace.append(point.copy())

    return tuple(point) if isinstance(initial_point, tuple) else point

def stochasticGradientDescent(gradient_function, initial_point, learning_rate, iterations):
    """
    Implements the stochastic gradient descent optimization algorithm.

    Parameters:
    - gradient_function: A function that computes the gradient at a given point.
    - initial_point: The starting point for the optimization.
    - learning_rate: The step size for each iteration.
    - iterations: The number of iterations to perform.

    Returns:
    - The optimized point after performing stochastic gradient descent.
    """
    if not callable(gradient_function):
        raise ValueError("Gradient function must be callable.")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, numbers.Real)
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("Learning rate must be a positive finite number.")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
        raise ValueError("Iterations must be a non-negative integer.")
    if not isinstance(initial_point, (list, tuple)) or not initial_point:
        raise ValueError("Initial point must be a non-empty list or tuple.")

    point = list(initial_point)
    for coordinate in point:
        if isinstance(coordinate, numbers.Real):
            if isinstance(coordinate, bool) or not math.isfinite(coordinate):
                raise ValueError("Initial point must contain finite numbers.")
        elif not isinstance(coordinate, (list, tuple)) or not coordinate:
            raise ValueError("Data points must be numbers or non-empty lists/tuples.")

    for _ in range(iterations):
        index = random.randrange(len(point))
        sampled_point = point[index]
        gradient = gradient_function(sampled_point)

        if isinstance(sampled_point, numbers.Real) and not isinstance(sampled_point, bool):
            if (
                isinstance(gradient, (list, tuple))
                and len(gradient) == 1
            ):
                gradient = gradient[0]
            if (
                isinstance(gradient, bool)
                or not isinstance(gradient, numbers.Real)
                or not math.isfinite(gradient)
            ):
                raise ValueError("Gradient function must return a finite number.")
            point[index] = sampled_point - learning_rate * gradient
        else:
            if not isinstance(sampled_point, (list, tuple)):
                raise ValueError("Data points must be numbers or lists/tuples of numbers.")
            if not isinstance(gradient, (list, tuple)) or len(gradient) != len(sampled_point):
                raise ValueError("Gradient function must return one value per coordinate.")
            updated = []
            for coordinate, component in zip(sampled_point, gradient):
                if (
                    isinstance(coordinate, bool)
                    or not isinstance(coordinate, numbers.Real)
                    or not math.isfinite(coordinate)
                    or
                    isinstance(component, bool)
                    or not isinstance(component, numbers.Real)
                    or not math.isfinite(component)
                ):
                    raise ValueError("Data points and gradients must contain finite numbers.")
                next_coordinate = coordinate - learning_rate * component
                if not math.isfinite(next_coordinate):
                    raise ValueError("Stochastic gradient descent produced a non-finite point.")
                updated.append(next_coordinate)
            point[index] = tuple(updated) if isinstance(sampled_point, tuple) else updated

    return tuple(point) if isinstance(initial_point, tuple) else point

def decisionTreeLearning(data, labels, depth=0, max_depth=None):
    """
    Implements a generic decision tree learning algorithm.

    Parameters:
    - data: A list of feature vectors (list of lists).
    - labels: A list of labels corresponding to the feature vectors.
    - depth: The current depth of the tree (used for recursion).
    - max_depth: The maximum depth of the tree (optional).

    Returns:
    - A decision tree represented as a nested dictionary.
    """
    return _cart_classification_tree(data, labels, depth, max_depth)

import numpy as np

class DecisionStump:
    """
    A simple weak learner: decision stump (one-level decision tree).
    It classifies based on a single feature threshold.
    """
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1
        self.alpha = None  # Weight assigned by AdaBoost

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X[:, self.feature_index] < self.threshold] = -1
        else:
            predictions[X[:, self.feature_index] >= self.threshold] = -1
        return predictions

class AdaBoost:
    """
    AdaBoost implementation using decision stumps as weak learners.
    """
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.stumps = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights equally
        w = np.full(n_samples, 1 / n_samples)

        for _ in range(self.n_estimators):
            stump = DecisionStump()
            min_error = float('inf')

            # Find best stump for current weights
            for feature_idx in range(n_features):
                feature_values = X[:, feature_idx]
                thresholds = np.unique(feature_values)
                for threshold in thresholds:
                    for polarity in [1, -1]:
                        predictions = np.ones(n_samples)
                        if polarity == 1:
                            predictions[feature_values < threshold] = -1
                        else:
                            predictions[feature_values >= threshold] = -1

                        # Calculate weighted error
                        error = np.sum(w[y != predictions])

                        if error < min_error:
                            min_error = error
                            stump.polarity = polarity
                            stump.threshold = threshold
                            stump.feature_index = feature_idx

            # Compute alpha (learner weight)
            epsilon = 1e-10  # Avoid division by zero
            stump.alpha = 0.5 * np.log((1 - min_error) / (min_error + epsilon))

            # Update weights
            predictions = stump.predict(X)
            w *= np.exp(-stump.alpha * y * predictions)
            w /= np.sum(w)  # Normalize

            self.stumps.append(stump)

    def predict(self, X):
        # Aggregate predictions from all stumps
        stump_preds = [stump.alpha * stump.predict(X) for stump in self.stumps]
        final_pred = np.sign(np.sum(stump_preds, axis=0))
        return final_pred

import numpy as np

def gradient_boosting(X, y, n_estimators=10, learning_rate=0.1, max_depth=3):
    from sklearn.tree import DecisionTreeRegressor

    # Normalize for stability
    X = np.float32(X)
    y = np.float32(y)

    # Initial prediction (mean)
    predictions = np.full(X.shape[0], np.mean(y))

    for i in range(n_estimators):
        # Compute residuals (negative gradient for squared loss)
        residuals = y - predictions

        # Fit weak learner to residuals
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=0)
        tree.fit(X, residuals)

        # Update predictions
        predictions += learning_rate * tree.predict(X)

    return predictions

import numpy as np

# -----------------------------
# Simple CART Regression Tree
# -----------------------------
class RegressionTree:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        # Stop conditions
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return np.mean(y)

        # Find best split
        best_feat, best_thresh, best_loss = None, None, float('inf')
        for feat_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feat_idx])
            for thresh in thresholds:
                left_mask = X[:, feat_idx] <= thresh
                right_mask = ~left_mask
                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue
                loss = self._mse(y[left_mask]) + self._mse(y[right_mask])
                if loss < best_loss:
                    best_feat, best_thresh, best_loss = feat_idx, thresh, loss

        if best_feat is None:
            return np.mean(y)

        # Split recursively
        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask
        return {
            'feature': best_feat,
            'threshold': best_thresh,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }

    def _mse(self, y):
        return np.var(y) * len(y)

    def predict_row(self, row, node):
        if not isinstance(node, dict):
            return node
        if row[node['feature']] <= node['threshold']:
            return self.predict_row(row, node['left'])
        else:
            return self.predict_row(row, node['right'])

    def predict(self, X):
        return np.array([self.predict_row(row, self.tree) for row in X])


# -----------------------------
# Simplified XGBoost Model
# -----------------------------
class XGBoostScratch:
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3, min_samples_split=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.base_prediction = None

    def fit(self, X, y):
        # Initial prediction is mean of y
        self.base_prediction = np.mean(y)
        y_pred = np.full(y.shape, self.base_prediction)

        for _ in range(self.n_estimators):
            # Compute residuals (negative gradient for MSE loss)
            residuals = y - y_pred

            # Fit regression tree to residuals
            tree = RegressionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, residuals)
            self.trees.append(tree)

            # Update predictions
            y_pred += self.learning_rate * tree.predict(X)

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.base_prediction)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

class isolationTree:
    def __init__(self, depth=0, max_depth=None):
        self.depth = depth
        self.max_depth = max_depth
        self.split_attr = None
        self.split_value = None
        self.left = None
        self.right = None
        self.size = 0

    def fit(self, X):
        self.size = len(X)

        # Stop conditions: max depth reached or only one sample
        if self.depth >= self.max_depth or self.size <= 1:
            return

        # Randomly choose an attribute (column)
        self.split_attr = random.randint(0, X.shape[1] - 1)

        # Randomly choose a split value between min and max of that attribute
        col_values = X[:, self.split_attr]
        min_val, max_val = col_values.min(), col_values.max()

        if min_val == max_val:  # No split possible
            return

        self.split_value = random.uniform(min_val, max_val)

        # Split data into left and right subsets
        left_mask = col_values < self.split_value
        right_mask = ~left_mask

        self.left = isolationTree(depth=self.depth + 1, max_depth=self.max_depth)
        self.left.fit(X[left_mask])

        self.right = isolationTree(depth=self.depth + 1, max_depth=self.max_depth)
        self.right.fit(X[right_mask])

    def path_length(self, x):
        # Path length for a single sample
        if self.split_attr is None or self.left is None or self.right is None:
            # External node: use c(n) adjustment
            return self.depth + c_factor(self.size)
        if x[self.split_attr] < self.split_value:
            return self.left.path_length(x)
        else:
            return self.right.path_length(x)


# ------------------------------
# Helper: Average Path Length c(n)
# ------------------------------
def c_factor(n):
    """Average path length of unsuccessful search in Binary Search Tree."""
    if n <= 1:
        return 0
    return 2 * (math.log(n - 1) + 0.5772156649) - (2 * (n - 1) / n)


# ------------------------------
# Isolation Forest Definition
# ------------------------------
class IsolationForest:
    def __init__(self, n_trees=100, sample_size=256):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.trees = []

    def fit(self, X):
        self.trees = []
        max_depth = math.ceil(math.log2(self.sample_size))
        for _ in range(self.n_trees):
            # Random subsample without replacement
            sample_indices = np.random.choice(len(X), self.sample_size, replace=False)
            sample = X[sample_indices]
            tree = isolationTree(depth=0, max_depth=max_depth)
            tree.fit(sample)
            self.trees.append(tree)

    def anomaly_score(self, X):
        scores = []
        for x in X:
            path_lengths = [tree.path_length(x) for tree in self.trees]
            avg_path_length = np.mean(path_lengths)
            score = 2 ** (-avg_path_length / c_factor(self.sample_size))
            scores.append(score)
        return np.array(scores)

    def predict(self, X, threshold=0.5):
        scores = self.anomaly_score(X)
        return np.where(scores >= threshold, 1, 0)  # 1 = anomaly, 0 = normal

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)


class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        """
        Initialize the Linear SVM classifier.
        :param learning_rate: Step size for gradient descent.
        :param lambda_param: Regularization strength.
        :param n_iters: Number of iterations for training.
        """
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Train the SVM classifier using gradient descent.
        :param X: Feature matrix (shape: [n_samples, n_features]).
        :param y: Labels (-1 or 1).
        """
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        # Ensure labels are -1 or 1
        y_ = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    # Only regularization term
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Regularization + hinge loss gradient
                    self.w -= self.lr * (2 * self.lambda_param * self.w - y_[idx] * x_i)
                    self.b -= self.lr * (-y_[idx])

    def predict(self, X):
        """
        Predict class labels for samples in X.
        :param X: Feature matrix.
        :return: Predicted labels (-1 or 1).
        """
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx)

import numpy as np
from scipy.spatial.distance import cdist

def kmeans(X, k, max_iterations=100, tolerance=1e-8):
    """
    K-Means clustering from scratch using NumPy.
    
    Parameters:
    - X: numpy array of shape (n_samples, n_features)
    - k: number of clusters
    - max_iterations: maximum number of iterations
    - tolerance: stop when centroid shift < tolerance
    
    Returns:
    - labels: cluster labels for each point
    - centroids: final centroids
    """
    # Step 1: Randomly initialize centroids
    centroids = X[np.random.choice(len(X), k, replace=False)].copy()
    
    for _ in range(max_iterations):
        # Step 2: Compute distances from each point to all centroids
        distances = cdist(X, centroids, metric='euclidean')
        
        # Step 3: Assign each point to nearest centroid
        labels = np.array([np.argmin(distances[i]) for i in range(len(X))])
        
        # Step 4: Update centroids as mean of assigned points
        new_centroids = np.vstack([
            X[(labels == cluster_idx).nonzero()[0]].mean(axis=0)
            for cluster_idx in range(k)
        ])
        
        # Check convergence
        if np.linalg.norm(new_centroids - centroids) <= tolerance:
            centroids = new_centroids
            break
        
        centroids = new_centroids
    
    return labels, centroids

def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def merge_clusters(distance_matrix, labels):
    # Find smallest distance > 0
    idx = np.triu_indices(len(distance_matrix), k=1)
    min_val = np.min(distance_matrix[idx])
    row, col = np.where(distance_matrix == min_val)
    # Merge clusters
    labels[labels == col] = row
    labels[labels > col] -= 1
    # Remove merged cluster from matrix
    dist = np.delete(distance_matrix, col, axis=0)
    dist = np.delete(dist, col, axis=1)
    return dist, labels

def agglomerative_clustering(X, method='average'):
    n = X.shape[0]
    # Initialize distance matrix
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            D[i, j] = euclidean_distance(X[i], X[j])
            D[j, i] = D[i, j]
    labels = np.arange(n)
    merges = []
    step = 1
    while step < n:
        # Find closest clusters
        idx = np.triu_indices(len(D), k=1)
        min_val = np.min(D[idx])
        row, col = np.where(D == min_val)
        # Merge
        labels[labels == col] = row
        labels[labels > col] -= 1
        merges.append((row, col, min_val))
        # Remove merged cluster
        D = np.delete(D, col, axis=0)
        D = np.delete(D, col, axis=1)
        step += 1
    return labels, merges



def _validate_cluster_input(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not len(X) or not X.shape[1]:
        raise ValueError("X must be a non-empty two-dimensional array.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite numbers.")
    return X


def _validate_n_clusters(n_clusters, n_samples):
    if isinstance(n_clusters, bool) or not isinstance(n_clusters, int) or not 1 <= n_clusters <= n_samples:
        raise ValueError("n_clusters must be an integer between 1 and the number of samples.")


def agglomartiveClustering(X, n_clusters=1, method="average"):
    """
    Bottom-up hierarchical clustering.

    Every point starts as its own cluster; the two closest clusters are merged
    repeatedly until n_clusters remain.

    Parameters:
    - X: Array of shape (n_samples, n_features).
    - n_clusters: Number of clusters to stop at.
    - method: Linkage — 'single' (min), 'complete' (max) or 'average' distance.

    Returns:
    - labels: Cluster index (0..n_clusters-1) for each sample.
    - merges: List of (cluster_a, cluster_b, distance) in merge order. Original
      points are ids 0..n-1; the cluster created by merge i gets id n + i.
    """
    X = _validate_cluster_input(X)
    _validate_n_clusters(n_clusters, len(X))
    if method not in {"single", "complete", "average"}:
        raise ValueError("method must be 'single', 'complete', or 'average'.")

    point_distances = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    clusters = {index: [index] for index in range(len(X))}
    merges = []
    next_id = len(X)

    while len(clusters) > n_clusters:
        ids = list(clusters)
        best_pair, best_distance = None, math.inf
        for i, first_id in enumerate(ids):
            for second_id in ids[i + 1:]:
                block = point_distances[np.ix_(clusters[first_id], clusters[second_id])]
                if method == "single":
                    distance = block.min()
                elif method == "complete":
                    distance = block.max()
                else:
                    distance = block.mean()
                if distance < best_distance:
                    best_pair, best_distance = (first_id, second_id), distance
        first_id, second_id = best_pair
        clusters[next_id] = clusters.pop(first_id) + clusters.pop(second_id)
        merges.append((first_id, second_id, float(best_distance)))
        next_id += 1

    labels = np.empty(len(X), dtype=int)
    for label, members in enumerate(clusters.values()):
        labels[members] = label
    return labels, merges


def DivisiveClustering(X, n_clusters=2, max_iterations=100, tolerance=1e-8):
    """
    Top-down hierarchical clustering (bisecting 2-means).

    All points start in one cluster; the cluster with the largest spread is
    repeatedly split in two with 2-means until n_clusters exist.

    Parameters:
    - X: Array of shape (n_samples, n_features).
    - n_clusters: Number of clusters to produce.
    - max_iterations: Max 2-means iterations per split.
    - tolerance: Centroid movement below which a split is considered converged.

    Returns:
    - labels: Cluster index for each sample.
    """
    X = _validate_cluster_input(X)
    _validate_n_clusters(n_clusters, len(X))
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer.")
    if isinstance(tolerance, bool) or not isinstance(tolerance, numbers.Real) or tolerance < 0:
        raise ValueError("tolerance must be a non-negative number.")

    clusters = [np.arange(len(X))]
    while len(clusters) < n_clusters:
        # Pick the cluster with the largest sum of squared errors that can be split.
        candidates = [
            (float(np.sum((X[members] - X[members].mean(axis=0)) ** 2)), index)
            for index, members in enumerate(clusters)
            if len(members) > 1
        ]
        if not candidates:
            break
        _, split_index = max(candidates)
        members = clusters.pop(split_index)
        points = X[members]

        # Seed 2-means with the point farthest from the mean and the point farthest from that.
        first = int(np.argmax(np.linalg.norm(points - points.mean(axis=0), axis=1)))
        second = int(np.argmax(np.linalg.norm(points - points[first], axis=1)))
        centers = points[[first, second]].copy()

        assignment = np.zeros(len(points), dtype=int)
        for _ in range(max_iterations):
            distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
            assignment = np.argmin(distances, axis=1)
            new_centers = centers.copy()
            for side in (0, 1):
                if np.any(assignment == side):
                    new_centers[side] = points[assignment == side].mean(axis=0)
            shift = np.max(np.linalg.norm(new_centers - centers, axis=1))
            centers = new_centers
            if shift <= tolerance:
                break

        left, right = members[assignment == 0], members[assignment == 1]
        if not len(left) or not len(right):
            # Identical points: fall back to an even split so progress is guaranteed.
            half = len(members) // 2
            left, right = members[:half], members[half:]
        clusters.extend((left, right))

    labels = np.empty(len(X), dtype=int)
    for label, members in enumerate(clusters):
        labels[members] = label
    return labels


def DBSCAN(X, eps=0.5, min_samples=5):
    """
    Density-Based Spatial Clustering of Applications with Noise.

    A point is a core point if at least min_samples points (including itself)
    lie within eps of it. Clusters grow outward from core points; points that
    are not reachable from any core point are noise.

    Parameters:
    - X: Array of shape (n_samples, n_features).
    - eps: Neighbourhood radius.
    - min_samples: Minimum neighbourhood size for a core point.

    Returns:
    - labels: Cluster index for each sample, with -1 for noise.
    """
    X = _validate_cluster_input(X)
    if isinstance(eps, bool) or not isinstance(eps, numbers.Real) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a positive finite number.")
    if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 1:
        raise ValueError("min_samples must be a positive integer.")

    distances = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    neighbors = [np.flatnonzero(row <= eps) for row in distances]
    is_core = np.array([len(n) >= min_samples for n in neighbors])

    UNVISITED, NOISE = -2, -1
    labels = np.full(len(X), UNVISITED, dtype=int)
    cluster = 0
    for point in range(len(X)):
        if labels[point] != UNVISITED:
            continue
        if not is_core[point]:
            labels[point] = NOISE  # may later be claimed as a border point
            continue
        labels[point] = cluster
        queue = list(neighbors[point])
        while queue:
            candidate = queue.pop()
            if labels[candidate] == NOISE:
                labels[candidate] = cluster  # border point
            if labels[candidate] != UNVISITED:
                continue
            labels[candidate] = cluster
            if is_core[candidate]:
                queue.extend(neighbors[candidate])
        cluster += 1
    return labels

def affinity_propagation(S, max_iter=200, damping=0.5, convergence_iter=15):
    """
    Affinity Propagation clustering from scratch.
    
    Parameters:
        S (ndarray): Similarity matrix (NxN), where S[i, j] is the similarity between points i and j.
        max_iter (int): Maximum number of iterations.
        damping (float): Damping factor between 0.5 and 1 to avoid oscillations.
        convergence_iter (int): Number of iterations with unchanged exemplars to declare convergence.
    
    Returns:
        cluster_centers_indices (list): Indices of cluster centers.
        labels (ndarray): Cluster label for each point.
    """
    N = S.shape[0]
    # Responsibility matrix
    R = np.zeros((N, N))
    # Availability matrix
    A = np.zeros((N, N))
    
    # Track convergence
    exemplars_history = []
    
    for iteration in range(max_iter):
        # Update responsibilities
        for i in range(N):
            for k in range(N):
                # Max over all other k' except k
                max_val = np.max(A[i, :] + S[i, :])
                max_val2 = np.max(np.delete(A[i, :] + S[i, :], k))
                R[i, k] = damping * R[i, k] + (1 - damping) * (S[i, k] - max_val2)
        
        # Update availabilities
        for k in range(N):
            for i in range(N):
                if i == k:
                    A[k, k] = damping * A[k, k] + (1 - damping) * np.sum(np.maximum(R[:, k], 0) - np.maximum(R[k, k], 0))
                else:
                    A[i, k] = damping * A[i, k] + (1 - damping) * min(0, R[k, k] + np.sum(np.maximum(R[np.arange(N) != i, k], 0)))
        
        # Check convergence
        E = (A + R) > 0
        exemplars = np.where(np.diag(E))[0]
        exemplars_history.append(exemplars)
        
        if iteration >= convergence_iter:
            if np.all(exemplars_history[-convergence_iter] == exemplars):
                break
    
    # Assign labels
    if len(exemplars) == 0:
        labels = np.zeros(N, dtype=int)
        cluster_centers_indices = []
    else:
        labels = np.array([exemplars[np.argmax(S[i, exemplars])] for i in range(N)])
        cluster_centers_indices = exemplars.tolist()
    
    return cluster_centers_indices, labels


class MeanShift:
    def __init__(self, radius=4, max_iter=10):
        self.radius = radius
        self.max_iter = max_iter

    def fit(self, data):
        centroids = {i: data[i].copy() for i in range(len(data))}
        for _ in range(self.max_iter):
            new_centroids = []
            for c in centroids.values():
                neighbors = [
                    x for x in data
                    if np.linalg.norm(x - c) < self.radius
                ]
                if neighbors:
                    new_c = np.mean(neighbors, axis=0)
                    new_centroids.append(new_c)
                else:
                    new_centroids.append(c)
            centroids = {i: np.array(new_centroids[i]) for i in range(len(new_centroids))}
            if np.allclose(list(centroids.values()), list(centroids.values())[:-1], equal_nan=True):
                break
        return centroids

def bellmanFord(vertices, edges, source):
    dist = [float('inf')] * vertices
    dist[source] = 0

    # Relax all edges V - 1 times
    for _ in range(vertices - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    # Check for negative-weight cycles
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            raise ValueError("Negative weight cycle detected")

    return dist

from collections import deque

def topologicalSort(graph):
    indegree = {}

    for node in graph:
        indegree[node] = indegree.get(node, 0)

        for neighbor in graph[node]:
            indegree[neighbor] = indegree.get(neighbor, 0) + 1

    queue = deque()

    for node in indegree:
        if indegree[node] == 0:
            queue.append(node)

    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbor in graph.get(node, []):
            indegree[neighbor] -= 1

            if indegree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(indegree):
        raise ValueError("Graph contains a cycle")

    return result

def minimumSpanningTree(vertices, edges):
    parent = list(range(vertices))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        parent[find(a)] = find(b)

    edges.sort(key=lambda edge: edge[2])

    mst = []
    total_weight = 0

    for u, v, weight in edges:
        if find(u) != find(v):
            union(u, v)
            mst.append((u, v, weight))
            total_weight += weight

    return {
        "edges": mst,
        "total_weight": total_weight
    }

def FloydWarshall(matrix):
    dist = [row[:] for row in matrix]

    n = len(dist)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if (
                    dist[i][k] != float('inf')
                    and dist[k][j] != float('inf')
                ):
                    dist[i][j] = min(
                        dist[i][j],
                        dist[i][k] + dist[k][j]
                    )

    return dist

import heapq

def huffmanCompression(text):
    if not text:
        return {}

    freq = {}

    for char in text:
        freq[char] = freq.get(char, 0) + 1

    heap = [[count, [char, ""]] for char, count in freq.items()]

    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        for pair in left[1:]:
            pair[1] = "0" + pair[1]

        for pair in right[1:]:
            pair[1] = "1" + pair[1]

        heapq.heappush(
            heap,
            [left[0] + right[0]] + left[1:] + right[1:]
        )

    codes = {}

    for char, code in heap[0][1:]:
        codes[char] = code

    encoded = ""

    for char in text:
        encoded += codes[char]

    return {
        "codes": codes,
        "encoded": encoded
    }

def bTree():