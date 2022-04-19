import numpy as np

class ConnectionHistory(object):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.allConnections = []
        self.global_innovation_count = 0

    def exists(self, n1, n2):
        for c in self.allConnections:
            if c.input_node.number == n1.number and c.output_node.number == n2.number:
                return c
        return None
        