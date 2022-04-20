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
            'edge' : 1.0,
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
            'node' : 0.5,
            'edge' : 0.25,
            'weight_perturb' : 0.4,
            'weight_set' : 0.1,
            'bias_perturb' : 0.3,
            'bias_set' : 0.1,
            're-enable': 0.2
        }

