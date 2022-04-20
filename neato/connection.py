class Connection(object):
    """A gene object representing a connection in the neural network."""
    def __init__(self, in_node, out_node, weight):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = True
        self.innovation = 0 

    def showConn(self):
        print("Connection innovation #", self.innovation, "Between", self.in_node.number, "in-layer",self.in_node.layer,"->", self.out_node.number, "out-layer",self.out_node.layer, "weight:", self.weight, "status:", self.enabled)