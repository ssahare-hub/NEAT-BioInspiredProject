class Node(object):
    """A gene object representing a node in the neural network."""
    def __init__(self, activation):
        self.output = 0
        self.bias = 0
        self.activation = activation
