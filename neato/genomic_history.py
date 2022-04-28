from typing import List
from .connection import *
from .node import *

class GenomicHistory(object):
    """ This object tracks all the connections added at the global level """
    def __init__(self, inputs: int, outputs: int, hidden_layers: int):
        self.inputs = inputs
        self.outputs = outputs
        self.allConnections: List[Connection] = []
        self.global_innovation_count = 0
        self.hidden_layers = hidden_layers
        self.allNodes: List[Node] = []

    def connection_exists(self, n1: Node, n2: Node):
        for c in self.allConnections:
            if c.in_node.number == n1.number and c.out_node.number == n2.number:
                return c
        return None

    def node_exists(self,n1: int):
        for node in self.allNodes:
            if node.number == n1:
                return node
        return None
