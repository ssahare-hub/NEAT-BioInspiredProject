import numpy as np

class node(object):
    def __init__(self, n, l,activation):
        self.output = 0 # used for forward propagation
        self.bias = 0
        self.activation = activation
        self.number = n
        self.layer = l
        self.inConnections = []