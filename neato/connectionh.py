import numpy as np

class ConnectionH(object):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.allConnections = []
        self.global_innov = 0

    def exists(self, n1, n2):
        for c in self.allConnections:
            if c.in_node.number == n1.number and c.out_node.number == n2.number:
                return c
        return None
        