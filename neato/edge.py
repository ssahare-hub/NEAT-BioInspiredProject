class Edge(object):
    """A gene object representing an edge in the neural network."""
    def __init__(self, weight):
        self.weight = weight
        self.enabled = True
