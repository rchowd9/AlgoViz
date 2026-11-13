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
    from collections import Counter
    import math 

    

    def euclidean_distance(point1, point2): 
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

    # Calculate distances from the query point to all points in the dataset
    distances = [
        (euclidean_distance(point, query_point), index, point, label)
        for index, (point, label) in enumerate(data)
    ]
    
    # Sort by distance and select the k nearest neighbors
    distances.sort(key=lambda x: x[0])
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
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        rows = X.tolist() if hasattr(X, "tolist") else list(X)
        labels = y.tolist() if hasattr(y, "tolist") else list(y)

        def build(indices, depth):
            counts = {}
            for index in indices:
                counts[labels[index]] = counts.get(labels[index], 0) + 1
            majority = max(counts, key=counts.get)
            if len(counts) == 1 or (self.max_depth is not None and depth >= self.max_depth):
                return {"value": majority}

            parent_impurity = 1 - sum((count / len(indices)) ** 2 for count in counts.values())
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
                return {"value": majority}
            feature, threshold, left, right = best_split
            return {
                "feature": feature,
                "threshold": threshold,
                "left": build(left, depth + 1),
                "right": build(right, depth + 1),
            }

        self.tree = build(list(range(len(rows))), 0)
        return self

    def predict(self, X):
        import numpy as np

        rows = X.tolist() if hasattr(X, "tolist") else list(X)

        def classify(row):
            node = self.tree
            while "value" not in node:
                node = node["left"] if float(row[node["feature"]]) <= node["threshold"] else node["right"]
            return node["value"]

        return np.asarray([classify(row) for row in rows])


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
        for _ in range(self.n_estimators):
            X_boot, y_boot = self._bootstrap_sample(X, y)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

    def predict(self, X):
        import numpy as np

        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0)  # regression: average; classification: majority vote

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
    from collections import Counter
    import numpy as np

    # Base case: if all labels are the same, return that label
    if len(set(labels)) == 1:
        return labels[0]

    # Base case: if no features left or max depth reached, return the most common label
    if len(data[0]) == 0 or (max_depth is not None and depth >= max_depth):
        return Counter(labels).most_common(1)[0][0]

    # Calculate entropy of current labels
    def entropy(labels):
        total = len(labels)
        counts = Counter(labels)
        return -sum((count / total) * np.log2(count / total) for count in counts.values())

    current_entropy = entropy(labels)

    # Find the best feature to split on
    best_gain = 0
    best_feature = None
    best_splits = None

    num_features = len(data[0])
    
    for feature_index in range(num_features):
        # Split data based on feature value
        splits = {}
        for i, row in enumerate(data):
            feature_value = row[feature_index]
            if feature_value not in splits:
                splits[feature_value] = []
            splits[feature_value].append(labels[i])

        # Calculate weighted entropy after the split
        weighted_entropy = sum((len(split) / len(labels)) * entropy(split) for split in splits.values())
        
        # Calculate information gain
        gain = current_entropy - weighted_entropy
        
        if gain > best_gain:
            best_gain = gain
            best_feature = feature_index
            best_splits = splits

    # If no gain, return the most common label
    if best_gain == 0:
        return Counter(labels).most_common(1)[0][0]

    # Create a subtree for each split
    tree = {best_feature: {}}

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
    from collections import Counter
    import numpy as np

    # Base case: if all labels are the same, return that label
    if len(set(labels)) == 1:
        return labels[0]

    # Base case: if no features left or max depth reached, return the most common label
    if len(data[0]) == 0 or (max_depth is not None and depth >= max_depth):
        return Counter(labels).most_common(1)[0][0]

    # Find the best feature to split on using Gini impurity
    def gini_impurity(labels):
        total = len(labels)
        counts = Counter(labels)
        return 1 - sum((count / total) ** 2 for count in counts.values())

    current_gini = gini_impurity(labels)

    best_gain = 0
    best_feature = None
    best_splits = None

def gradientDescent(gradient_function, initial_point, learning_rate, iterations):
    """
    Implements the gradient descent optimization algorithm.

    Parameters:
    - gradient_function: A function that computes the gradient at a given point.
    - initial_point: The starting point for the optimization.
    - learning_rate: The step size for each iteration.
    - iterations: The number of iterations to perform.

    Returns:
    - The optimized point after performing gradient descent.
    """
    point = initial_point

    for _ in range(iterations):
        gradient = gradient_function(point)
        point = [p - learning_rate * g for p, g in zip(point, gradient)]

    return point

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
    point = initial_point

    for _ in range(iterations):
        # Randomly sample a data point (or mini-batch) for stochastic update
        sampled_point = random.choice(point)  # Assuming point is a list of data points
        gradient = gradient_function(sampled_point)
        point = [p - learning_rate * g for p, g in zip(point, gradient)]

    return point

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
    from collections import Counter
    import numpy as np

    # Base case: if all labels are the same, return that label
    if len(set(labels)) == 1:
        return labels[0]

    # Base case: if no features left or max depth reached, return the most common label
    if len(data[0]) == 0 or (max_depth is not None and depth >= max_depth):
        return Counter(labels).most_common(1)[0][0]

    # Find the best feature to split on using Gini impurity
    def gini_impurity(labels):
        total = len(labels)
        counts = Counter(labels)
        return 1 - sum((count / total) ** 2 for count in counts.values())

    current_gini = gini_impurity(labels)

    best_gain = 0
    best_feature = None
    best_splits = None

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
from sklearn.tree import DecisionTreeRegressor

def gradient_boosting(X, y, n_estimators=10, learning_rate=0.1, max_depth=3):
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

# Example usage
np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3 * X[:, 0]**2 + 0.05 * np.random.randn(100)

predictions = gradient_boosting(X, y, n_estimators=10, learning_rate=0.1, max_depth=3)
