from typing import List
from connection import *
from node import *


class ConnectionHistory(object):
    def __init__(self, inputs: int, outputs: int, hidden_layers: int):
        self.inputs = inputs
        self.outputs = outputs
        self.allConnections: List[Connection] = []
        self.global_innovation_count = 0
        self.hidden_layers = hidden_layers

    def exists(self, n1: Node, n2: Node):
        for c in self.allConnections:
            if c.in_node.number == n1.number and c.out_node.number == n2.number:
                return c
        return None
