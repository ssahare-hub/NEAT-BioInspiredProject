class Connection(object):
    """A gene object representing a connection in the neural network."""
    def __init__(self, in_node, out_node, weight):
        self.input_node = in_node
        self.output_node = out_node
        self.weight = weight
        self.enabled = True
        self.innovation = 0 

    def showConn(self):
        print("Connection innovation #", self.innovation, "Between", self.input_node.number, "in-layer",self.input_node.layer,"->", self.output_node.number, "out-layer",self.output_node.layer, "weight:", self.weight, "status:", self.enabled)