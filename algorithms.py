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