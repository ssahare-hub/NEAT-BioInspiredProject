import random
import math
import copy
from genome import *

def genomic_crossover(a, b):
    """Breed two genomes and return the child. Matching genes
    are inherited randomly, while disjoint genes are inherited
    from the fitter parent.
    """
    # Template genome for child
    child = Genome(a.connection_history, a._default_activation)
    a_in = set(a._connections)
    b_in = set(b._connections)

    # Inherit homologous gene from a random parent
    for i in a_in & b_in:
        parent = random.choice([a, b])
        child._connections[i] = copy.deepcopy(parent._connections[i])

    # Inherit disjoint/excess genes from fitter parent
    if a._fitness > b._fitness:
        for i in a_in - b_in:
            child._connections[i] = copy.deepcopy(a._connections[i])
    else:
        for i in b_in - a_in:
            child._connections[i] = copy.deepcopy(b._connections[i])
    
    # Calculate max node
    child._total_nodes = 0
    for innovation in child._connections:
        in_node = child._connections[innovation].in_node.number
        out_node = child._connections[innovation].out_node.number
        current_max = max(in_node, out_node)
        child._total_nodes = max(child._total_nodes, current_max)
    child._total_nodes += 1

    # Inherit nodes
    for n in range(child._total_nodes):
        inherit_from = []
        if n in a._nodes:
            inherit_from.append(a)
        if n in b._nodes:
            inherit_from.append(b)

        random.shuffle(inherit_from)
        parent = max(inherit_from, key=lambda p: p._fitness)
        child._nodes[n] = copy.deepcopy(parent._nodes[n])

    child.reset()
    return child


class Specie(object):
    """A specie represents a set of genomes whose genomic distances 
    between them fall under the Brain's delta threshold.
    """
    def __init__(self, max_fitness_history, *members):
        self._members = list(members)
        self._fitness_history = []
        self._fitness_sum = 0
        self._max_fitness_history = max_fitness_history

    def breed(self, mutation_probabilities, breed_probabilities):
        """Return a child as a result of either a mutated clone
        or crossover between two parent genomes.
        """
        # Either mutate a clone or breed two random genomes
        population = list(breed_probabilities.keys())
        probabilities= [breed_probabilities[k] for k in population]
        choice = random.choices(population, weights=probabilities)[0]

        if choice == "asexual" or len(self._members) == 1:
            child = random.choice(self._members).clone()
            child.mutate(mutation_probabilities)
        elif choice == "sexual":
            (mom, dad) = random.sample(self._members, 2)
            child = genomic_crossover(mom, dad)
        
        # child.mutate(mutation_probabilities)
        return child

    def update_fitness(self):
        """Update the adjusted fitness values of each genome 
        and the historical fitness."""
        for g in self._members:
            g._adjusted_fitness = g._fitness/len(self._members)

        self._fitness_sum = sum([g._adjusted_fitness for g in self._members])
        self._fitness_history.append(self._fitness_sum)
        if len(self._fitness_history) > self._max_fitness_history:
            self._fitness_history.pop(0)

    def cull_genomes(self, fittest_only):
        """Exterminate the weakest genomes per specie."""
        self._members.sort(key=lambda g: g._fitness, reverse=True)
        if fittest_only:
            # Only keep the winning genome
            remaining = 1
        else:
            # Keep top 25%
            remaining = int(math.ceil(0.25*len(self._members)))

        self._members = self._members[:remaining]

    def get_best(self):
        """Get the member with the highest fitness score."""
        return max(self._members, key=lambda g: g._fitness)

    def can_progress(self):
        """Determine whether species should survive the culling."""
        n = len(self._fitness_history)
        avg = sum(self._fitness_history) / n
        return avg > self._fitness_history[0] or n < self._max_fitness_history
