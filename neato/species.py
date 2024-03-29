import random
import math
import copy
from .genome import *
from .node import *

class Species(object):
    """A species represents individuals within the same niche 
    which is determined by the genomic distance being below δ_threshold
    """

    def __init__(self, max_fitness_history: int, *members):
        self._members = list(members)
        self._fitness_history = []
        self._fitness_sum = 0
        self._max_fitness_history = max_fitness_history

    def reproduce(self, hyperparams: Hyperparameters) -> Node:
        """Genrate and return offspring (child) either through cloning or crossover 
        followed by mutation.
        """
        mutation_probabilities = hyperparams.mutation_probabilities
        breed_probabilities = hyperparams.reproduce_probabilities
        # Either mutate a clone or breed two random genomes
        population = list(breed_probabilities.keys())
        probabilities = [breed_probabilities[k] for k in population]
        choice = random.choices(population, weights=probabilities)[0]

        if choice == "asexual" or len(self._members) == 1:
            child = random.choice(self._members).clone()
            # child.mutate(hyperparams)
        elif choice == "sexual":
            (mom, dad) = random.sample(self._members, 2)
            child = genomic_crossover(mom, dad)    
            # if random.random() <= mutation_probabilities['mutate']:
            #     child.mutate(hyperparams)

        if random.random() <= mutation_probabilities['mutate']:
                child.mutate(hyperparams)
        return child

    def update_fitness(self) -> None:
        """Calculates the adjusted fitness of genomes in a species"""
        for g in self._members:
            g._adjusted_fitness = g._fitness/len(self._members)

        self._fitness_sum = sum([g._adjusted_fitness for g in self._members])
        self._fitness_history.append(self._fitness_sum)
        if len(self._fitness_history) > self._max_fitness_history:
            self._fitness_history.pop(0)

    def ranked_selection(self, survival_percentage: float) -> None:
        """Rank genome in species and eliminate the least fit individuals"""
        self._members.sort(key=lambda g: g._fitness, reverse=True)

        # keep the top percentage
        survivor_count = int(math.ceil(survival_percentage*len(self._members)))
        self._members = self._members[:survivor_count]

    def get_best(self) -> Node:
        """Get the member with the highest fitness score."""
        return max(self._members, key=lambda g: g._fitness)

    def can_progress(self) -> bool:
        """Determine whether species should survive the culling."""
        n = len(self._fitness_history)
        avg = sum(self._fitness_history) / n
        return avg >= self._fitness_history[0] or n <= self._max_fitness_history


def genomic_crossover(a: Genome, b: Genome) -> Genome:
    """
    Genomic crossover of two networks
    """
    # Template genome for child
    child = Genome(a.genmoic_history, a._default_activation)
    a_in = set(a._connections)
    b_in = set(b._connections)
    matching_connections = a_in & b_in

    # Inherit homologous gene from a random parent
    for i in matching_connections:
        parent = random.choice([a, b])
        child._connections[i] = copy.deepcopy(parent._connections[i])

    # Inherit disjoint/excess genes from fitter parent
    disjoint_connections = a_in - b_in
    parent = a

    if a._fitness <= b._fitness:
        disjoint_connections = b_in - a_in
        parent = b

    for i in disjoint_connections:
        child._connections[i] = copy.deepcopy(parent._connections[i])

    # Calculate total nodes
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

        parent = max(inherit_from, key=lambda p: p._fitness)
        child._nodes[n] = copy.deepcopy(parent._nodes[n])

    # reset the output values of all nodes
    child.reset()
    return child