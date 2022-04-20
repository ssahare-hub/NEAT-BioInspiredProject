import numpy as np

class ConnectionHistory(object):
    def __init__(self, inputs, outputs, hidden_layers):
        self.inputs = inputs
        self.outputs = outputs
        self.allConnections = []
        self.global_innovation_count = 0
        self.hidden_layers = hidden_layers

    def exists(self, n1, n2):
        for c in self.allConnections:
            if c.in_node.number == n1.number and c.out_node.number == n2.number:
                return c
        return None
        