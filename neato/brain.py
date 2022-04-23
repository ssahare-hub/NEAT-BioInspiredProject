import pickle
import multiprocessing as mp
from .genome import *
from .hyperparameters import *
from .connection_history import *
from .species import *


def genomic_distance(a: Node, b: Node, distance_weights: dict):
    """Calculate the genomic distance between two genomes."""
    a_connections = set(a._connections)
    b_connections = set(b._connections)

    # Does not distinguish between disjoint and excess
    matching_connections = a_connections & b_connections
    disjoint_connections = (
        a_connections - b_connections) | (b_connections - a_connections)
    num_max_connections = len(max(a_connections, b_connections, key=len))
    num_min_nodes = min(a._total_nodes, b._total_nodes)

    weight_diff = 0
    for i in matching_connections:
        weight_diff += abs(a._connections[i].weight - b._connections[i].weight)

    bias_diff = 0
    for i in range(num_min_nodes):
        bias_diff += abs(a._nodes[i].bias - b._nodes[i].bias)

    t1 = distance_weights['disjoint_connections'] * \
        len(disjoint_connections)/num_max_connections
    t2 = distance_weights['matching_connections'] * \
        len(matching_connections)/num_max_connections
    t3 = distance_weights['weight'] * weight_diff/len(matching_connections)
    t4 = distance_weights['bias'] * bias_diff/num_min_nodes
    return t1 + t3 + t4  # + t2


class Brain(object):
    """Base class for a 'brain' that learns through the evolution
    of a population of genomes.
    """

    def __init__(self, inputs: int, outputs: int, hidden_layers: int, population: int = 100, hyperparams: Hyperparameters = Hyperparameters()):
        self._inputs = inputs
        self._outputs = outputs
        self._hidden_layers = hidden_layers
        self._connection_history = ConnectionHistory(
            inputs, outputs, hidden_layers)

        self._species = []
        self._population = population

        # Hyper-parameters
        self._hyperparams = hyperparams

        self._generation = 0
        self._current_species = 0
        self._current_genome = 0

        self._global_best = None

    def generate(self):
        """Generate the initial population of genomes."""
        for _ in range(self._population):
            g = Genome(self._connection_history,
                       self._hyperparams.default_activation)
            g.createNetwork()
            self.classify_genome(g)
        print("generating genome for population")

        # Set the initial best genome
        self._global_best = self._species[0]._members[0]

    def classify_genome(self, genome: Genome):
        """Classify genomes into species via the genomic
        distance algorithm.
        """
        if not self._species:
            # Empty population
            self._species.append(Species(
                    self._hyperparams.max_fitness_history, genome
                )
            )
        else:
            # Compare genome against representative s[0] in each Species
            for s in self._species:
                rep = s._members[0]
                dist = genomic_distance(
                    genome, rep, self._hyperparams.distance_weights
                )
                if dist <= self._hyperparams.delta_threshold:
                    s._members.append(genome)
                    return

            # Doesn't fit with any other specie, create a new one
            self._species.append(Species(
                    self._hyperparams.max_fitness_history, genome
                )
            )

    def update_fittest(self):
        """Update the highest fitness score of the whole population."""
        top_performers = [s.get_best() for s in self._species]
        current_top = max(top_performers, key=lambda g: g._fitness)

        if current_top._fitness > self._global_best._fitness:
            self._global_best = current_top.clone()

    def evolve(self):
        """Evolve the population by eliminating the poorest performing
        genomes and repopulating with mutated children, prioritizing
        the most promising species.
        """
        global_fitness_sum = 0
        for s in self._species:
            s.update_fitness()
            global_fitness_sum += s._fitness_sum

        if global_fitness_sum == 0:
            # No progress, mutate everybody
            for s in self._species:
                for g in s._members:
                    g.mutate(self._hyperparams.mutation_probabilities)
        else:
            # Only keep the species with potential to improve
            surviving_species = []
            for s in self._species:
                if s.can_progress():
                    surviving_species.append(s)
            self._species = surviving_species

            # Eliminate lowest performing genomes per Species
            for s in self._species:
                s.cull_genomes(False)

            # Repopulate
            for i, s in enumerate(self._species):
                ratio = s._fitness_sum/global_fitness_sum
                diff = self._population - self.get_population()
                offspring = int(round(ratio * diff))
                for j in range(offspring):
                    self.classify_genome(
                        s.breed(
                            self._hyperparams.mutation_probabilities,
                            self._hyperparams.breed_probabilities
                        )
                    )

            # No species survived
            # Repopulate using mutated minimal structures and global best
            if not self._species:
                for i in range(self._population):
                    if i % 3 == 0:
                        g = self._global_best.clone()
                    else:
                        g = Genome(self._inputs, self._outputs,
                                   self._hyperparams.default_activation)
                        g.generate()
                    g.mutate(self._hyperparams.mutation_probabilities)
                    self.classify_genome(g)

        self._generation += 1

    def should_evolve(self):
        """Determine if the system should continue to evolve
        based on the maximum fitness and generation count.
        """
        self.update_fittest()
        fit = self._global_best._fitness <= self._hyperparams.max_fitness
        print(self._global_best._fitness,self._hyperparams.max_fitness)
        end = self._generation != self._hyperparams.max_generations

        return fit and end

    def next_iteration(self):
        """Call after every evaluation of individual genomes to
        progress training.
        """
        s = self._species[self._current_species]
        if self._current_genome < len(s._members)-1:
            self._current_genome += 1
        else:
            if self._current_species < len(self._species)-1:
                self._current_species += 1
                self._current_genome = 0
            else:
                # Evolve to the next generation
                self.evolve()
                self._current_species = 0
                self._current_genome = 0

    def evaluate_parallel(self, evaluator: Callable, *args, **kwargs):
        """Evaluate the entire population on separate processes
        to progress training. The evaluator function must take a Genome
        as its first parameter and return a numerical fitness score.

        Any global state passed to the evaluator is copied and will not
        be modified at the parent process.
        """
        max_proc = max(mp.cpu_count()-1, 1)
        pool = mp.Pool(processes=max_proc)

        results = {}
        for i in range(len(self._species)):
            for j in range(len(self._species[i]._members)):
                results[(i, j)] = pool.apply_async(
                    evaluator,
                    args=[self._species[i]._members[j]]+list(args),
                    kwds=kwargs
                )

        for key in results:

            print(results[key])
            genome = self._species[key[0]]._members[key[1]]
            genome.set_fitness(results[key])

        pool.close()
        pool.join()
        self.evolve()

    def get_fittest(self):
        """Return the genome with the highest global fitness score."""
        return self._global_best

    def get_population(self):
        """Return the true (calculated) population size."""
        return sum([len(s._members) for s in self._species])

    def get_current(self):
        """Get the current genome for evaluation."""
        s = self._species[self._current_species]
        return s._members[self._current_genome]

    def get_current_species(self):
        """Get index of current species being evaluated."""
        return self._current_species

    def get_current_genome(self):
        """Get index of current genome being evaluated."""
        return self._current_genome

    def get_generation(self):
        """Get the current generation number of this population."""
        return self._generation

    def get_species(self):
        """Get the list of species and their respective member genomes."""
        return self._species

    def save(self, filename: str):
        """Save an instance of the population to disk."""
        with open(filename+'.neat', 'wb') as _out:
            pickle.dump(self, _out, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename):
        """Return an instance of a population from disk."""
        with open(filename+'.neat', 'rb') as _in:
            return pickle.load(_in)
