from typing import Callable, List
from .connection import *

class Node(object):
    """
    Node object carries information about the activation, bias, layer, incoming connections and output.
    """
    def __init__(self, n: int, l: int, activation: Callable):
        self.output = 0 
        self.bias = 0
        self.activation: Callable = activation
        self.number = n
        self.layer = l
        self.inConnections: List[Connection] = []
    # to read the node object when used within a print statement
    def __repr__(self):
        return str(self.number)
