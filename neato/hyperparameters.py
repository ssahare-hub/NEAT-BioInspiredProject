import math


def sigmoid(x: float):
    """Return the S-Curve activation of x."""
    return 1/(1+math.exp(-x))


def tanh(x: float):
    """Wrapper function for hyperbolic tangent activation."""
    return math.tanh(x)


def LReLU(x: float):
    """Leaky ReLU function for x"""
    if x >= 0:
        return x
    else:
        return 0.01 * x


class Hyperparameters(object):
    """Hyperparameter settings for the Brain object."""

    def __init__(self):
        self.delta_threshold = 1.5
        self.distance_weights = {
            'disjoint_connections': 1.0,
            'matching_connections': 1,
            'weight': 1.0,
            'bias': 1.0
        }
        self.default_activation = sigmoid

        self.max_fitness = float(500)
        self.max_generations = float('inf')
        self.max_fitness_history = 30
        self.fitness_offset = 0

        self.breed_probabilities = {
            'asexual': 0.5,
            'sexual': 0.5
        }
        self.mutation_probabilities = {
            'mutate': 0.05,
            'node': 0.05,
            'connection': 0.05,
            'weight_perturb': 0.04,
            'weight_set': 0.01,
            'bias_perturb': 0.03,
            'bias_set': 0.01,
            're-enable': 0.009
        }
