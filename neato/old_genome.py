import random
import copy
import itertools
from old_node import Node
from edge import Edge

class OldGenome(object):
    """Base class for a standard genome used by the NEAT algorithm."""
    def __init__(self, inputs, outputs, default_activation):
        # Nodes
        self._inputs = inputs
        self._outputs = outputs

        self._unhidden = inputs+outputs
        self._max_node = inputs+outputs

        # Structure
        self._edges = {} # (i, j) : Edge
        self._nodes = {} # NodeID : Node

        self._default_activation = default_activation

        # Performance
        self._fitness = 0
        self._adjusted_fitness = 0

    def generate(self):
        """Generate the neural network of this genome with minimal
        initial topology, i.e. (no hidden nodes). Call on genome
        creation.
        """
        # Minimum initial topology, no hidden layer
        for n in range(self._max_node):
            self._nodes[n] = Node(self._default_activation)

        for i in range(self._inputs):
            for j in range(self._inputs, self._unhidden):
                self.add_edge(i, j, random.uniform(-1, 1))
                
    def forward(self, inputs):
        """Evaluate inputs and calculate the outputs of the
        neural network via the forward propagation algorithm.
        """
        if len(inputs) != self._inputs:
            raise ValueError("Incorrect number of inputs.")

        # Set input values
        for i in range(self._inputs):
            self._nodes[i].output = inputs[i]
        
        # Generate backward-adjacency list 
        _from = {}
        for n in range(self._max_node):
            _from[n] = []

        for (i, j) in self._edges:
            if not self._edges[(i, j)].enabled:
                continue
            _from[j].append(i)

        # Calculate output values for each node
        ordered_nodes = itertools.chain(
            range(self._unhidden, self._max_node),
            range(self._inputs, self._unhidden)
        )
        for j in ordered_nodes:
            ax = 0
            for i in _from[j]:
                ax += self._edges[(i, j)].weight * self._nodes[i].output
                # print(i,j)

            node = self._nodes[j]
            node.output = node.activation(ax + node.bias)
            # print(j,node.output)
        
        return [self._nodes[n].output for n in range(self._inputs, self._unhidden)]

    def mutate(self, probabilities):
        """Randomly mutate the genome to initiate variation."""
        if self.is_disabled():
            self.add_enabled()

        population = list(probabilities.keys())
        weights = [probabilities[k] for k in population]
        choice = random.choices(population, weights=weights)[0]

        if choice == "node":
            self.add_node()
            print("add node")
        elif choice == "edge":
            print("add connection")
            (i, j) = self.random_pair()
            self.add_edge(i, j, random.uniform(-1, 1))
        elif choice == "weight_perturb" or choice == "weight_set":
            self.shift_weight(choice)
            print(choice)
        elif choice == "bias_perturb" or choice == "bias_set":
            self.shift_bias(choice)
            print(choice)

        self.reset()

    def add_node(self):
        """Add a new node between a randomly selected edge,
        disabling the parent edge.
        """
        enabled = [k for k in self._edges if self._edges[k].enabled]
        (i, j) = random.choice(enabled)
        edge = self._edges[(i, j)]
        edge.enabled = False

        new_node = self._max_node
        self._max_node += 1
        self._nodes[new_node] = Node(self._default_activation)

        self.add_edge(i, new_node, 1.0)
        self.add_edge(new_node, j, edge.weight)

    def add_edge(self, i, j, weight):
        """Add a new connection between existing nodes."""
        if (i, j) in self._edges:
            self._edges[(i, j)].enabled = True
        else:
            self._edges[(i, j)] = Edge(weight)
            
    def add_enabled(self):
        """Re-enable a random disabled edge."""
        disabled = [e for e in self._edges if not self._edges[e].enabled]

        if len(disabled) > 0:
            self._edges[random.choice(disabled)].enabled = True
        
    def shift_weight(self, type):
        """Randomly shift, perturb, or set one of the edge weights."""
        e = random.choice(list(self._edges.keys()))
        if type == "weight_perturb":
            self._edges[e].weight += random.uniform(-1, 1)
        elif type == "weight_set":
            self._edges[e].weight = random.uniform(-1, 1)

    def shift_bias(self, type):
        """Randomly shift, perturb, or set the bias of an incoming edge."""
        # Select only nodes in the hidden and output layer
        n = random.choice(range(self._inputs, self._max_node))
        if type == "bias_perturb":
            self._nodes[n].bias += random.uniform(-1, 1)
        elif type == "bias_set":
            self._nodes[n].bias = random.uniform(-1, 1)

    def random_pair(self):
        """Generate random nodes (i, j) such that:
        1. i is not an output
        2. j is not an input
        3. i != j
        """
        i = random.choice([n for n in range(self._max_node) if not self.is_output(n)])
        j_list = [n for n in range(self._max_node) if not self.is_input(n) and n != i]

        if not j_list:
            j = self._max_node
            self.add_node()
        else:
            j = random.choice(j_list)

        return (i, j)

    def is_input(self, n):
        """Determine if the node id is an input."""
        return 0 <= n < self._inputs

    def is_output(self, n):
        """Determine if the node id is an output."""
        return self._inputs <= n < self._unhidden

    def is_hidden(self, n):
        """Determine if the node id is hidden."""
        return self._unhidden <= n < self._max_node

    def is_disabled(self):
        """Determine if all of its genes are disabled."""
        return all(self._edges[i].enabled == False for i in self._edges)

    def get_fitness(self):
        """Return the fitness of the genome."""
        return self._fitness

    def get_nodes(self):
        """Get the nodes of the network."""
        return self._nodes.copy()

    def get_edges(self):
        """Get the network's edges."""
        return self._edges.copy()

    def get_num_nodes(self):
        """Get the number of nodes in the network."""
        return self._max_node

    def set_fitness(self, score):
        """Set the fitness score of this genome."""
        self._fitness = score

    def reset(self):
        """Reset the genome's internal state."""
        for n in range(self._max_node):
            self._nodes[n].output = 0
        self._fitness = 0

    def clone(self):
        """Return a clone of the genome.
        """
        return copy.deepcopy(self)


