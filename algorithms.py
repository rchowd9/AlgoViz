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

def gradient_boosting(X, y, n_estimators=10, learning_rate=0.1, max_depth=3):
    """Fit shallow regression trees to squared-error residuals and return fitted values."""
    features = np.asarray(X, dtype=float)
    targets = np.asarray(y, dtype=float)
    if features.ndim != 2 or not len(features) or features.shape[1] == 0:
        raise ValueError("X must be a non-empty two-dimensional array.")
    if targets.ndim != 1 or len(targets) != len(features):
        raise ValueError("y must contain one target for each row in X.")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(targets)):
        raise ValueError("X and y must contain only finite numbers.")
    if isinstance(n_estimators, bool) or not isinstance(n_estimators, int) or n_estimators < 1:
        raise ValueError("n_estimators must be a positive integer.")
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0:
        raise ValueError("max_depth must be a non-negative integer.")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, numbers.Real) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    predictions = np.full(len(targets), targets.mean(), dtype=float)
    for _ in range(n_estimators):
        tree = RegressionTree(max_depth=max_depth)
        tree.fit(features, targets - predictions)
        predictions += learning_rate * tree.predict(features)
    return predictions

# -----------------------------
# Simple CART Regression Tree
# -----------------------------
class RegressionTree:
    def __init__(self, max_depth=3, min_samples_split=2):
        if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0:
            raise ValueError("max_depth must be a non-negative integer.")
        if isinstance(min_samples_split, bool) or not isinstance(min_samples_split, int) or min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer of at least 2.")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        features = np.asarray(X, dtype=float)
        targets = np.asarray(y, dtype=float)
        if features.ndim != 2 or not len(features) or features.shape[1] == 0:
            raise ValueError("X must be a non-empty two-dimensional array.")
        if targets.ndim != 1 or len(targets) != len(features):
            raise ValueError("y must contain one target for each row in X.")
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(targets)):
            raise ValueError("X and y must contain only finite numbers.")
        self.tree = self._build_tree(features, targets, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split or np.all(y == y[0]):
            return np.mean(y)

        best_feat, best_thresh, best_loss = None, None, float('inf')
        for feat_idx in range(X.shape[1]):
            values = np.unique(X[:, feat_idx])
            thresholds = (values[:-1] + values[1:]) / 2
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

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask
        return {
            "feature": best_feat,
            "threshold": best_thresh,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[right_mask], y[right_mask], depth + 1),
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
        if self.tree is None:
            raise ValueError("The regression tree must be fitted before prediction.")
        features = np.asarray(X, dtype=float)
        if features.ndim != 2 or features.shape[1] == 0:
            raise ValueError("X must be a two-dimensional array.")
        return np.asarray([self.predict_row(row, self.tree) for row in features])


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
        features = np.asarray(X, dtype=float)
        targets = np.asarray(y, dtype=float)
        if features.ndim != 2 or not len(features) or features.shape[1] == 0:
            raise ValueError("X must be a non-empty two-dimensional array.")
        if targets.ndim != 1 or len(targets) != len(features):
            raise ValueError("y must contain one target for each row in X.")
        if isinstance(self.n_estimators, bool) or not isinstance(self.n_estimators, int) or self.n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer.")
        if isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, numbers.Real) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        self.base_prediction = float(targets.mean())
        y_pred = np.full(len(targets), self.base_prediction)
        self.trees = []

        for _ in range(self.n_estimators):
            tree = RegressionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(features, targets - y_pred)
            self.trees.append(tree)
            y_pred += self.learning_rate * tree.predict(features)
        return self

    def predict(self, X):
        if self.base_prediction is None:
            raise ValueError("The model must be fitted before prediction.")
        features = np.asarray(X, dtype=float)
        if features.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")
        y_pred = np.full(len(features), self.base_prediction)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(features)
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
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")
        self.size = len(X)

        if self.max_depth is None:
            self.max_depth = max(1, math.ceil(math.log2(max(self.size, 2))))
        if self.depth >= self.max_depth or self.size <= 1 or X.shape[1] == 0:
            return

        splittable = [
            index for index in range(X.shape[1])
            if X[:, index].min() < X[:, index].max()
        ]
        if not splittable:
            return
        self.split_attr = random.choice(splittable)
        col_values = X[:, self.split_attr]
        min_val, max_val = col_values.min(), col_values.max()
        self.split_value = random.uniform(min_val, max_val)
        left_mask = col_values < self.split_value
        right_mask = ~left_mask
        if not np.any(left_mask) or not np.any(right_mask):
            self.split_attr = None
            self.split_value = None
            return

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

def kmeans(X, k, max_iterations=100, tolerance=1e-8):
    """
    Cluster rows of X with Lloyd's k-means algorithm.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a non-empty two-dimensional array.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite numbers.")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(X):
        raise ValueError("k must be an integer between 1 and the number of samples.")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer.")
    if not isinstance(tolerance, numbers.Real) or isinstance(tolerance, bool) or tolerance < 0:
        raise ValueError("tolerance must be a non-negative number.")

    centroids = X[random.sample(range(len(X)), k)].copy()
    labels = np.zeros(len(X), dtype=int)
    for _ in range(max_iterations):
        distances = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(distances, axis=1)
        new_centroids = centroids.copy()
        for cluster in range(k):
            members = X[new_labels == cluster]
            if len(members):
                new_centroids[cluster] = members.mean(axis=0)
            else:
                # Re-seed empty clusters at a point far from its assigned centroid.
                errors = distances[np.arange(len(X)), new_labels]
                new_centroids[cluster] = X[int(np.argmax(errors))]
        converged = np.array_equal(labels, new_labels) and np.allclose(
            centroids, new_centroids, atol=tolerance, rtol=0
        )
        labels, centroids = new_labels, new_centroids
        if converged:
            break
    return labels, centroids

def euclidean_distance(p1, p2):
    first = np.asarray(p1, dtype=float)
    second = np.asarray(p2, dtype=float)
    if first.shape != second.shape:
        raise ValueError("Points must have matching dimensions.")
    return float(np.linalg.norm(first - second))

def merge_clusters(distance_matrix, labels):
    distances = np.asarray(distance_matrix, dtype=float)
    merged_labels = np.asarray(labels).copy()
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("Distance matrix must be square.")
    if len(merged_labels) != len(distances) or len(distances) < 2:
        raise ValueError("At least two matching clusters are required.")
    upper = np.triu_indices(len(distances), k=1)
    row, col = min(
        zip(upper[0], upper[1]),
        key=lambda pair: distances[pair[0], pair[1]],
    )
    merged_labels[merged_labels == merged_labels[col]] = merged_labels[row]
    reduced = np.delete(np.delete(distances, col, axis=0), col, axis=1)
    return reduced, merged_labels


def agglomerative_clustering(X, method="average", n_clusters=1):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not len(X) or not X.shape[1]:
        raise ValueError("X must be a non-empty two-dimensional array.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite numbers.")
    if method not in {"single", "complete", "average"}:
        raise ValueError("method must be 'single', 'complete', or 'average'.")
    if isinstance(n_clusters, bool) or not isinstance(n_clusters, int) or not 1 <= n_clusters <= len(X):
        raise ValueError("n_clusters must be between 1 and the number of samples.")

    point_distances = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    clusters = {index: [index] for index in range(len(X))}
    active = list(clusters)
    merges = []
    next_id = len(X)
    while len(active) > n_clusters:
        pair = None
        pair_distance = math.inf
        for i, first_id in enumerate(active):
            for second_id in active[i + 1:]:
                distances = [
                    point_distances[first, second]
                    for first in clusters[first_id]
                    for second in clusters[second_id]
                ]
                distance = (
                    min(distances) if method == "single"
                    else max(distances) if method == "complete"
                    else sum(distances) / len(distances)
                )
                if distance < pair_distance:
                    pair, pair_distance = (first_id, second_id), distance
        first_id, second_id = pair
        merges.append((first_id, second_id, float(pair_distance)))
        clusters[next_id] = clusters.pop(first_id) + clusters.pop(second_id)
        active = [cluster_id for cluster_id in active if cluster_id not in pair]
        active.append(next_id)
        next_id += 1

    labels = np.empty(len(X), dtype=int)
    for label, cluster_id in enumerate(active):
        labels[clusters[cluster_id]] = label
    return labels, merges


def agglomartiveClustering(X, n_clusters=1, method="average"):
    """Compatibility spelling for agglomerative clustering."""
    return agglomerative_clustering(X, method=method, n_clusters=n_clusters)


def divisive_clustering(X, n_clusters=2, max_iterations=100, tolerance=1e-8):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not len(X) or not X.shape[1]:
        raise ValueError("X must be a non-empty two-dimensional array.")
    if isinstance(n_clusters, bool) or not isinstance(n_clusters, int) or not 1 <= n_clusters <= len(X):
        raise ValueError("n_clusters must be between 1 and the number of samples.")
    clusters = [list(range(len(X)))]
    while len(clusters) < n_clusters:
        splittable = [
            (float(np.sum(np.var(X[indices], axis=0))), index)
            for index, indices in enumerate(clusters)
            if len(indices) > 1
        ]
        if not splittable:
            break
        _, cluster_index = max(splittable)
        indices = clusters.pop(cluster_index)
        points = X[indices]
        centers = points[[int(np.argmax(np.linalg.norm(points - points.mean(axis=0), axis=1))), 0]]
        if np.array_equal(centers[0], centers[1]):
            centers[1] = points[-1]
        local_labels = np.zeros(len(points), dtype=int)
        for _ in range(max_iterations):
            distances = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            updated_labels = np.argmin(distances, axis=1)
            updated_centers = centers.copy()
            for label in (0, 1):
                members = points[updated_labels == label]
                if len(members):
                    updated_centers[label] = members.mean(axis=0)
            if np.array_equal(local_labels, updated_labels) and np.allclose(
                centers, updated_centers, atol=tolerance, rtol=0
            ):
                local_labels = updated_labels
                centers = updated_centers
                break
            local_labels, centers = updated_labels, updated_centers
        left = [indices[index] for index in range(len(indices)) if local_labels[index] == 0]
        right = [indices[index] for index in range(len(indices)) if local_labels[index] == 1]
        if not left or not right:
            left, right = indices[:len(indices) // 2], indices[len(indices) // 2:]
        clusters.extend((left, right))
    result = np.empty(len(X), dtype=int)
    for label, indices in enumerate(clusters):
        result[indices] = label
    return result


def DivisiveClustering(X, n_clusters=2, max_iterations=100, tolerance=1e-8):
    """Compatibility spelling for divisive clustering."""
    return divisive_clustering(X, n_clusters, max_iterations, tolerance)


def DBSCAN(X, eps=0.5, min_samples=5):
    """Density-based clustering; noise samples receive label -1."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not len(X) or not X.shape[1]:
        raise ValueError("X must be a non-empty two-dimensional array.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite numbers.")
    if isinstance(eps, bool) or not isinstance(eps, numbers.Real) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a positive finite number.")
    if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 1:
        raise ValueError("min_samples must be a positive integer.")

    distances = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    neighbors = [np.flatnonzero(distances[index] <= eps).tolist() for index in range(len(X))]
    labels = np.full(len(X), -1, dtype=int)
    visited = np.zeros(len(X), dtype=bool)
    cluster = 0
    for point in range(len(X)):
        if visited[point]:
            continue
        visited[point] = True
        if len(neighbors[point]) < min_samples:
            continue
        labels[point] = cluster
        seeds = list(neighbors[point])
        queued = set(seeds)
        cursor = 0
        while cursor < len(seeds):
            candidate = seeds[cursor]
            cursor += 1
            if not visited[candidate]:
                visited[candidate] = True
                if len(neighbors[candidate]) >= min_samples:
                    for neighbor in neighbors[candidate]:
                        if neighbor not in queued:
                            queued.add(neighbor)
                            seeds.append(neighbor)
            if labels[candidate] == -1:
                labels[candidate] = cluster
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
    similarities = np.asarray(S, dtype=float).copy()
    if similarities.ndim != 2 or similarities.shape[0] != similarities.shape[1] or not len(similarities):
        raise ValueError("S must be a non-empty square similarity matrix.")
    if not np.all(np.isfinite(similarities)):
        raise ValueError("S must contain only finite numbers.")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")
    if (
        isinstance(damping, bool) or not isinstance(damping, numbers.Real)
        or not math.isfinite(damping) or not 0 <= damping < 1
    ):
        raise ValueError("damping must be a number in [0, 1).")
    if isinstance(convergence_iter, bool) or not isinstance(convergence_iter, int) or convergence_iter < 1:
        raise ValueError("convergence_iter must be a positive integer.")

    size = len(similarities)
    responsibilities = np.zeros_like(similarities)
    availabilities = np.zeros_like(similarities)
    history = []
    for _ in range(max_iter):
        values = availabilities + similarities
        best_indices = np.argmax(values, axis=1)
        best_values = values[np.arange(size), best_indices]
        second_values = np.partition(values, -2, axis=1)[:, -2] if size > 1 else best_values
        competing = np.broadcast_to(second_values[:, None], values.shape).copy()
        competing[np.arange(size), best_indices] = best_values
        updated_responsibilities = similarities - competing
        responsibilities = damping * responsibilities + (1 - damping) * updated_responsibilities

        positive = np.maximum(responsibilities, 0)
        column_sums = positive.sum(axis=0) - np.diag(positive)
        updated_availabilities = np.minimum(
            0,
            responsibilities.diagonal()[None, :] + column_sums[None, :] - positive,
        )
        np.fill_diagonal(updated_availabilities, column_sums)
        availabilities = damping * availabilities + (1 - damping) * updated_availabilities

        exemplars = np.diag(availabilities + responsibilities) > 0
        history.append(exemplars.copy())
        if len(history) >= convergence_iter and all(
            np.array_equal(previous, history[-1])
            for previous in history[-convergence_iter:]
        ) and np.any(exemplars):
            break

    centers = np.flatnonzero(exemplars)
    if not len(centers):
        centers = np.asarray([int(np.argmax(np.diag(similarities)))])
    labels = np.asarray(
        [centers[np.argmax(similarities[index, centers])] for index in range(size)],
        dtype=int,
    )
    return centers.tolist(), labels


class MeanShift:
    def __init__(self, radius=4, max_iter=10, tolerance=1e-3):
        self.radius = radius
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, data):
        X = np.asarray(data, dtype=float)
        if X.ndim != 2 or not len(X) or not X.shape[1]:
            raise ValueError("data must be a non-empty two-dimensional array.")
        if not np.all(np.isfinite(X)):
            raise ValueError("data must contain only finite numbers.")
        if isinstance(self.radius, bool) or not isinstance(self.radius, numbers.Real) or self.radius <= 0:
            raise ValueError("radius must be a positive number.")
        if isinstance(self.max_iter, bool) or not isinstance(self.max_iter, int) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")
        if not isinstance(self.tolerance, numbers.Real) or self.tolerance < 0:
            raise ValueError("tolerance must be non-negative.")

        modes = []
        for point in X:
            center = point.copy()
            for _ in range(self.max_iter):
                neighbors = X[np.linalg.norm(X - center, axis=1) <= self.radius]
                shifted = neighbors.mean(axis=0)
                if np.linalg.norm(shifted - center) <= self.tolerance:
                    center = shifted
                    break
                center = shifted
            modes.append(center)

        centers = []
        for mode in modes:
            if not any(np.linalg.norm(mode - center) <= self.radius / 2 for center in centers):
                centers.append(mode)
        self.cluster_centers_ = np.asarray(centers)
        distances = np.linalg.norm(X[:, None, :] - self.cluster_centers_[None, :, :], axis=2)
        self.labels_ = np.argmin(distances, axis=1)
        return {index: center.copy() for index, center in enumerate(self.cluster_centers_)}

    def predict(self, data):
        if self.cluster_centers_ is None:
            raise ValueError("MeanShift must be fitted before prediction.")
        X = np.asarray(data, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError("data must have the same number of features used during fitting.")
        distances = np.linalg.norm(X[:, None, :] - self.cluster_centers_[None, :, :], axis=2)
        return np.argmin(distances, axis=1)
