from collections import defaultdict
import numpy as np
import random
import copy
import itertools
from node import Node
from connection import Connection


class Genome(object):
    def __init__(self, connection_history, hidden_layers, default_activation, willCreate):
        self.connection_history = connection_history
        self._inputs = connection_history.inputs
        self._outputs = connection_history.outputs
        self._unhidden = connection_history.inputs + connection_history.outputs

        self.default_activation = default_activation

        # Structure
        self.input_layer = 0
        self.output_layer = hidden_layers+1

        self.total_nodes = 0
        self.creation_rate = 1

        self._nodes = {}
        self._connections = {}

        # Performance
        self._fitness = 0
        self._adjusted_fitness = 0

        if willCreate:
            self.createNetwork()

    def createNetwork(self):
        # Create Input nodes
        for i in range(self._inputs):
            self._nodes[i] = Node(
                self.total_nodes, self.input_layer, self.default_activation)
            self.total_nodes += 1
        
        # Create Output nodes
        for i in range(self._inputs, self._inputs+self._outputs):
            self._nodes[i] = Node(
                self.total_nodes, self.output_layer, self.default_activation)
            self.total_nodes += 1

        # Add random connections between input and output nodes
        # NOTE: This might be redundant
        for i in range(self._inputs * self._outputs):
            if np.random.random() < self.creation_rate:
                # create a random connection
                self.add_connection()

        # create the minimal network by connecting all input nodes to all output nodes
        for i in range(self._inputs):
            for j in range(self._inputs, self._unhidden):
                self.add_connection(i, j, random.uniform(-1, 1))

    def add_connection(self, i=-1, j=-1, weight=random.uniform(-1, 1)):

        # NOTE: Random connection creation is unreliable
        if i == -1 & j == -1:
            n1 = self._nodes[np.random.randint(0, len(self._nodes))]
            n2 = self._nodes[np.random.randint(0, len(self._nodes))]

            # to assure that chosen node is not on outputlayer
            while n1.layer == self.output_layer:  
                n1 = self._nodes[np.random.randint(0, len(self._nodes))]

            # to assure second node not in a lower layer than first node
            while n2.layer == self.input_layer or n2.layer <= n1.layer:
                n2 = self._nodes[np.random.randint(0, len(self._nodes))]
        else:
            n1 = self._nodes[i]
            n2 = self._nodes[j]

        new_connection = Connection(n1, n2, weight)

        # NOTE: Checks if this connection exists in the connection history of all genomes
        old_connection = self.connection_history.exists(n1, n2)
        if old_connection:
            print("connection already exists in the connection history with innovation", old_connection.innovation)
            # if it does, assigns the same innovation number
            new_connection.innovation = old_connection.innovation
            if not self.exists(new_connection.innovation):
                self._connections[new_connection.innovation] = new_connection
                n2.inConnections.append(new_connection)
        else:
            # the new connection is not in the connection history
            # hence assign a new innovation number to this connection
            new_connection.innovation = self.connection_history.global_innovation_count
            self.connection_history.global_innovation_count += 1
            
            # Add it to dict of all connection 
            self._connections[new_connection.innovation] = new_connection
            
            # Add it to the connection history
            self.connection_history.allConnections.append(new_connection)  
            
            # add a new incoming connection to the node
            n2.inConnections.append(new_connection)

    # check if a connection with same innovation number exists in the genome
    def exists(self, innovation_number):
        for n in self._connections:
            if self._connections[n].innovation == innovation_number:
                return True
        return False

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
        _from = defaultdict(list)
        _innov_to_connections = defaultdict(int)

        for n in self._connections:
            connection = self._connections[n]
            if not connection.enabled:
                continue
            _from[connection.out_node].append(connection.in_node)
            _innov_to_connections[(connection.in_node, connection.out_node)] = connection.innovation
        

        # Calculate output values for each node\
        ordered_nodes = itertools.chain(
            range(self._unhidden, self.total_nodes),
            range(self._inputs, self._unhidden)
        )
        for j in ordered_nodes:
            ax = 0
            for i in _from[j]:
                innovation = _innov_to_connections[(i,j)]
                ax += self._connections[innovation].weight * self._nodes[i].output

            node = self._nodes[j]
            node.output = node.activation(ax + node.bias)

        return [self._nodes[n].output for n in range(self._inputs, self._unhidden)]

    def mutate(self, probabilities):
        """Randomly mutate the genome to initiate variation."""
        if self.is_disabled():
            self.add_enabled()

        potential_connections = [n for n in self._connections if self._connections[n].enabled and self._connections[n].in_node.layer+1 < self._connections[n].out_node.layer]
        choices = list(probabilities.keys())

        if not potential_connections:
            choices.remove("node")
            print("No node can be added!")

        weights = [probabilities[k] for k in choices]
        choice = random.choices(choices, weights=weights)[0]

        if choice == "node":
            print("add node")
            self.add_node(potential_connections)
        elif choice == "connection":
            (i, j) = self.random_pair()
            self.add_connection(i, j, random.uniform(-1, 1))
            print("add connection between:")
            # self._connections[(i, j)].showConn()
        elif choice == "weight_perturb" or choice == "weight_set":
            print(choice)
            self.shift_weight(choice)
        elif choice == "bias_perturb" or choice == "bias_set":
            print(choice)
            self.shift_bias(choice)
        elif choice == "re-enable":
            print('re enabling a random connection')
            self.add_enabled()
        self.reset()

    def add_node(self, potential_connections):
        """Add a new node between a randomly selected connection,
        disabling the parent connection.
        """
        n = random.choice(potential_connections)
        connection = self._connections[n]
        connection.enabled = False

        print(connection.in_node.layer + 1, connection.out_node.layer)
        new_node_layer = np.random.randint(
            connection.in_node.layer + 1, connection.out_node.layer)
        new_node = self.total_nodes
        self.total_nodes += 1
        # print(new_node,self.total_nodes)
        self._nodes[new_node] = Node(
            new_node, new_node_layer, self.default_activation)

        self.add_connection(connection.in_node.number, new_node, 1.0)
        self.add_connection(new_node, connection.out_node.number, connection.weight)

    def add_enabled(self):
        """Re-enable a random disabled connection."""
        disabled = [
            e for e in self._connections if not self._connections[e].enabled]

        if len(disabled) > 0:
            self._connections[random.choice(disabled)].enabled = True

    def shift_weight(self, type):
        """Randomly shift, perturb, or set one of the connection weights."""
        e = random.choice(list(self._connections.keys()))
        if type == "weight_perturb":
            self._connections[e].weight += random.uniform(-1, 1)
        elif type == "weight_set":
            self._connections[e].weight = random.uniform(-1, 1)

    def shift_bias(self, type):
        """Randomly shift, perturb, or set the bias of an incoming connection."""
        # Select only nodes in the hidden and output layer
        n = random.choice(range(self._inputs, self.total_nodes))
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
        if self.total_nodes == self._inputs+self._outputs:
            i = random.choice(
                [n for n in range(self.total_nodes) if not self.is_output(n)])
            j_list = [n for n in range(self.total_nodes) if not self.is_input(
                n) and n != i and self._nodes[n].layer > self._nodes[i].layer]

            if not j_list:
                j = self.total_nodes
                self.add_node()
            else:
                j = random.choice(j_list)
        else:
            innovation = random.choice([n for n in self._connections])
            while innovation in self._connections:
                i = random.choice(
                    [n for n in range(self.total_nodes) if not self.is_output(n)])
                j_list = [n for n in range(self.total_nodes) if not self.is_input(
                    n) and n != i and self._nodes[n].layer > self._nodes[i].layer]

                if not j_list:
                    j = self.total_nodes
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
        return self._unhidden <= n < self.total_nodes

    def is_disabled(self):
        """Determine if all of its genes are disabled."""
        return all(self._connections[i].enabled == False for i in self._connections)

    def get_fitness(self):
        """Return the fitness of the genome."""
        return self._fitness

    def get_nodes(self):
        """Get the nodes of the network."""
        return self._nodes.copy()

    def get_connections(self):
        """Get the network's connections."""
        return self._connections.copy()

    def get_num_nodes(self):
        """Get the number of nodes in the network."""
        return self.total_nodes

    def set_fitness(self, score):
        """Set the fitness score of this genome."""
        self._fitness = score

    def reset(self):
        """Reset the genome's internal state."""
        for n in range(self.total_nodes):
            self._nodes[n].output = 0
        self._fitness = 0

    def clone(self):
        """Return a clone of the genome.
        """
        return copy.deepcopy(self)
    # def add_node(self):
    #     # self.nodes.append(Node(self.total_nodes, np.random.randint(self.input_layer+1, self.output_layer),self.default_activation))
    #     self.nodes.append(Node(self.total_nodes, np.random.randint(self.input_layer+1, self.output_layer),self.default_activation))
    #     self.total_nodes += 1

    # def randomize(self):
    #     self.weight = np.random.random() * 4 - 2

    # def shift(self):
    #     self.weight += np.random.uniform(-0.2, 0.2)

    # def toggle(self):
    #     self.enabled = not self.enabled
