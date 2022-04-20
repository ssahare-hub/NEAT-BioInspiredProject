import math

def sigmoid(x):
    """Return the S-Curve activation of x."""
    return 1/(1+math.exp(-x))

def tanh(x):
    """Wrapper function for hyperbolic tangent activation."""
    return math.tanh(x)

def LReLU(x):
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
            'disjoint_connections' : 1.0,
            'matching_connections' : 1,
            'weight' : 1.0,
            'bias' : 1.0
        }
        self.default_activation = sigmoid

        self.max_fitness = float('inf')
        self.max_generations = float('inf')
        self.max_fitness_history = 30

        self.breed_probabilities = {
            'asexual' : 0.5,
            'sexual' : 0.5
        }
        self.mutation_probabilities = {
            'node' : 0.05,
            'edge' : 0.05,
            'weight_perturb' : 0.04,
            'weight_set' : 0.01,
            'bias_perturb' : 0.03,
            'bias_set' : 0.01,
            're-enable': 0.0005
        }

