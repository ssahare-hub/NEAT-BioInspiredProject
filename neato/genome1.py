import numpy as np
import random
import copy
import itertools
from node1 import node
from connection import Connection

class genome(object):
    def __init__(self, ch, hidden_layers, default_activation, create):
        self.ch = ch
        self._inputs = ch.inputs
        self._outputs = ch.outputs
        self._unhidden = ch.inputs + ch.outputs

        self._default_activation = default_activation

        # Structure 
        self.input_layer = 0
        self.output_layer = hidden_layers+1
        
        self._total_nodes = 0
        self.creation_rate = 1
        
        # self.nodes = []
        self._nodes = {}
        self._connections = {}
        # self._connections = []

        # Performance
        self._fitness = 0
        self._adjusted_fitness = 0
        
        if create:
            self.createNetwork()
           
    def createNetwork(self):
        for i in range(self._inputs):
            # self.nodes.append(node(self._total_nodes, self.input_layer, self._default_activation))
            self._nodes[i] = node(self._total_nodes, self.input_layer, self._default_activation)
            self._total_nodes += 1
            # print(self._total_nodes)
        for i in range(self._inputs, self._inputs+self._outputs):
            # self.nodes.append(node(self._total_nodes, self.output_layer, self._default_activation))
            self._nodes[i] = node(self._total_nodes, self.output_layer, self._default_activation)
            self._total_nodes += 1
            # print(self._nodes[i].number)
        
        
        for i in range(self._inputs * self._outputs):
            if np.random.random() < self.creation_rate:
                self.add_connection()
        
        for i in range(self._inputs):
            for j in range(self._inputs, self._unhidden):
                self.add_connection(i, j, random.uniform(-1, 1))
    
    def add_connection(self,i=-1,j=-1,weight=random.uniform(-1, 1)):

        # n1 = self.nodes[np.random.randint(0, len(self.nodes))]
        # n2 = self.nodes[np.random.randint(0, len(self.nodes))]
        if i == -1 & j ==-1:
            n1 = self._nodes[np.random.randint(0, len(self._nodes))]
            n2 = self._nodes[np.random.randint(0, len(self._nodes))]

            while n1.layer == self.output_layer: # to assure that chosen node is not on outputlayer
                # n1 = self.nodes[np.random.randint(0, len(self.nodes))]
                n1 = self._nodes[np.random.randint(0, len(self._nodes))]

            while n2.layer == self.input_layer or n2.layer <= n1.layer: # to assure second node not in a lower layer than first node
                # n2 = self.nodes[np.random.randint(0, len(self.nodes))]
                n2 = self._nodes[np.random.randint(0, len(self._nodes))]
        else:
            n1 = self._nodes[i]
            n2 = self._nodes[j]

        

        c = self.ch.exists(n1, n2)
        x = Connection(n1, n2, weight)
        x.showConn

        if c != None:
            print("c is not empty",c.innov)
            x.innov = c.innov
            if not self.exists(x.innov):
                # self._connections.append(x)
                self._connections[x.innov] = x
                n2.inConnections.append(x)
        else:
            x.innov = self.ch.global_innov
            self.ch.global_innov += 1
            # self._connections.append(x)
            self._connections[x.innov] = x
            self.ch.allConnections.append(x)#(x.copy())
            n2.inConnections.append(x)
        
    def exists(self, nn):
        # for c in self._connections:
        #     if c.innov == nn:
        #         return True
        # return False
        for n in self._connections:
            if self._connections[n].innov == nn:
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
        _from = {}
        for n in range(self._total_nodes):
            _from[n] = []

        for n in self._connections:
            if not self._connections[n].enabled:
                continue
            _from[self._connections[n].out_node.number].append(self._connections[n].out.number)

        # Calculate output values for each node
        ordered_nodes = itertools.chain(
            range(self._unhidden, self._total_nodes),
            range(self._inputs, self._unhidden)
        )
        for j in ordered_nodes:
            ax = 0
            for i in _from[j]:
                ax += self._connections[(i, j)].weight * self._nodes[i].output

            node = self._nodes[j]
            node.output = node.activation(ax + node.bias)
        
        return [self._nodes[n].output for n in range(self._inputs, self._unhidden)]

    def mutate(self, probabilities):
        """Randomly mutate the genome to initiate variation."""
        if self.is_disabled():
            self.add_enabled()

        population = list(probabilities.keys())
        weights = [probabilities[k] for k in population]
        choice = random.choices(population, weights=weights)[0]

        if choice == "node":
            print("add node")
            self.add_node()
        elif choice == "connection":
            (i, j) = self.random_pair()
            self.add_connection(i, j, random.uniform(-1, 1))
            print("add connection between:")
            # self._connections[(i,j)].showConn()
        elif choice == "weight_perturb" or choice == "weight_set":
            print(choice)
            self.shift_weight(choice)
        elif choice == "bias_perturb" or choice == "bias_set":
            print(choice)
            self.shift_bias(choice)

        self.reset()

    def add_node(self):
        """Add a new node between a randomly selected connection,
        disabling the parent connection.
        """
        enabled = [k for k in self._connections if self._connections[k].enabled and self._connections[k].in_node.layer+1 < self._connections[k].out_node.layer]
        k = random.choice(enabled)
        connection = self._connections[k]
        connection.enabled = False

        print(connection.in_node.layer +1, connection.out_node.layer)
        new_node_layer = np.random.randint(connection.in_node.layer +1, connection.out_node.layer)
        new_node = self._total_nodes
        self._total_nodes += 1
        # print(new_node,self._total_nodes)
        self._nodes[new_node] = node(new_node,new_node_layer,self._default_activation)

        self.add_connection(i, new_node, 1.0)
        self.add_connection(new_node, j, connection.weight)

    def add_enabled(self):
        """Re-enable a random disabled connection."""
        disabled = [e for e in self._connections if not self._connections[e].enabled]

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
        n = random.choice(range(self._inputs, self._total_nodes))
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
        if self._total_nodes == self._inputs+self._outputs:
            i = random.choice([n for n in range(self._total_nodes) if not self.is_output(n)])
            j_list = [n for n in range(self._total_nodes) if not self.is_input(n) and n != i and self._nodes[n].layer > self._nodes[i].layer]

            if not j_list:
                j = self._total_nodes
                self.add_node()
            else:
                j = random.choice(j_list)
        else:
            innov = random.choice([n for n in self._connections])
            while innov in self._connections:
                i = random.choice([n for n in range(self._total_nodes) if not self.is_output(n)])
                j_list = [n for n in range(self._total_nodes) if not self.is_input(n) and n != i and self._nodes[n].layer > self._nodes[i].layer]

                if not j_list:
                    j = self._total_nodes
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
        return self._unhidden <= n < self._total_nodes

    def is_disabled(self):
        """Determine if all of its genes are disabled."""
        return all(self._connections[i].enabled == False for i in self._connections)

    def get_innov(self):
        """Returns all innovation numbers for the established connections"""
        return [self._connections[i].innov for i in self._connections]

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
        return self._total_nodes

    def get_connection(self,innov):
        """Get connection index from innovation number"""
        n = [n for n in self._connections if self._connections[n].innov == innov]
        return n

    def set_fitness(self, score):
        """Set the fitness score of this genome."""
        self._fitness = score

    def reset(self):
        """Reset the genome's internal state."""
        for n in range(self._total_nodes):
            self._nodes[n].output = 0
        self._fitness = 0

    def clone(self):
        """Return a clone of the genome.
        """
        return copy.deepcopy(self)
    # def add_node(self):
    #     # self.nodes.append(node(self._total_nodes, np.random.randint(self.input_layer+1, self.output_layer),self._default_activation))
    #     self.nodes.append(node(self._total_nodes, np.random.randint(self.input_layer+1, self.output_layer),self._default_activation))
    #     self._total_nodes += 1
    
    # def randomize(self):
    #     self.weight = np.random.random() * 4 - 2    

    # def shift(self):
    #     self.weight += np.random.uniform(-0.2, 0.2)

    # def toggle(self):
    #     self.enabled = not self.enabled
