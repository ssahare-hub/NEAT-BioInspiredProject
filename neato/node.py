from typing import Callable, List
from .connection import *

class Node(object):
    def __init__(self, n: int, l: int, activation: Callable):
        self.output = 0  # used for forward propagation
        self.bias = 0
        self.activation: Callable = activation
        self.number = n
        self.layer = l
        self.inConnections: List[Connection] = []
    
    def __repr__(self):
        return str(self.number)
