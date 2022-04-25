from .node import *


class Connection(object):
    """A gene object representing a connection in the neural network."""

    def __init__(self, in_node: Node, out_node: Node, weight: float):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = True
        self.innovation = 0

    def showConn(self):
        print("Connection innovation #", self.innovation, "Between", self.in_node.number,"->", self.out_node.number, "weight:", self.weight, "status:", self.enabled)
        print("{:>10s} {:>14d} -> {:>14d}".format("Layers:",self.in_node.layer,self.out_node.layer))

    def __repr__(self):
        return(f"{self.in_node.number} -> {self.out_node.number}")