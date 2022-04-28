import os
import pickle
import multiprocessing as mp

from matplotlib import pyplot as plt
from .genome import *
from .hyperparameters import *
from .genomic_history import *
from .species import *
from tqdm import tqdm

class NeatO(object):
    """ 
    The main object which is responsible for creating genomes, evolution, speciation, culling, and reproduction.
    """

    def __init__(self, inputs: int, outputs: int, hidden_layers: int, population: int = 100, hyperparams: Hyperparameters = Hyperparameters()):
        self._inputs = inputs
        self._outputs = outputs
        self._hidden_layers = hidden_layers
        self._genomic_history = GenomicHistory(
            inputs, outputs, hidden_layers)

        self._species: List[Species] = []
        self._population = population
        self._network_history = []
        self._population_history = []      

        # Hyper-parameters
        self._hyperparams = hyperparams

        self._generation = 0
        self._current_species = 0
        self._current_genome = 0
        self._past_fitnesses = []
        self._global_best = None
        self._past_max_fitnesses = []
        self._current_fittest: Genome = None

    def initialize(self):
        """Generate the initial population of genomes."""
        for _ in range(self._population):
            g = Genome(self._genomic_history,
                       self._hyperparams.default_activation)
            g.create_network()
            self.speciation(g)
        # print("generating genome for population")

        # Set the initial best genome
        self._global_best = self._species[0]._members[0]

    def speciation(self, genome: Genome):
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

    def find_current_fittest(self):
        """Update the highest fitness score of the whole population."""
        top_performers = [s.get_best() for s in self._species]
        current_top = max(top_performers, key=lambda g: g._fitness)
        return current_top.clone()

    def get_current_fittest(self):
        return self._current_fittest
    
    def evolve(self):
        """Evolve the population by eliminating the poorest performing
        genomes and repopulating with mutated children, prioritizing
        the most promising species.
        """
        global_fitness_sum = 0
        # print('updating fitness fo species')
        for s in self._species:
            s.update_fitness()
            global_fitness_sum += s._fitness_sum

        # update the fittest genome of the generation
        self._current_fittest = self.find_current_fittest()
        
        if global_fitness_sum == 0:
            # print('No progress, mutating every genome')
            # No progress, mutate everybody
            for s in self._species:
                for g in s._members:
                    g.mutate(self._hyperparams)
        else:
            # Only keep the species with potential to improve
            # print('Collecting species with potential to improve')
            surviving_species = []
            for s in self._species:
                if s.can_progress():
                    surviving_species.append(s)
            self._species = surviving_species

            # print('Culling low performing genomes')
            # Rank and eliminate lowest performing genomes per Species
            for s in self._species:
                s.ranked_selection(self._hyperparams.survival_percentage)

            # print('Repopulating')
            # Repopulate
            Children = []
            for i, s in enumerate(self._species):
                ratio = s._fitness_sum/global_fitness_sum
                diff = self._population - self.get_population()
                # # offspring = int(round(ratio * diff))
                # offspring = int(round(ratio * self._population))
                # for j in range(offspring):
                #     self.speciation(
                #         s.breed(
                #             self._hyperparams
                #         )
                #     )

            #     # NOTE: to have a fixed population size instead
                offspring = int(round(ratio * self._population))
                
                
                for j in range(offspring):
                    Children.append(s.reproduce(self._hyperparams))

            self._species = []
            for child in Children:
                self.speciation(child)
            # end for changes to have a fixed population size

            # No species survived
            # Repopulate using mutated minimal structures and global best
            if not self._species:
                print('='*100)
                print('No species survived, repopulating!!')
                print('='*100)
                for i in range(self._population):
                    if i % 3 != 0:
                        g = self._global_best.clone()
                    else:
                        g = Genome(self._genomic_history,
                                   self._hyperparams.default_activation, 
                                   True)
                    g.mutate(self._hyperparams)
                    self.speciation(g)

        self._generation += 1

    def should_evolve(self):
        """Determine if the system should continue to evolve
        based on the maximum fitness and generation count.
        """
        self.update_fittest()
        fit = self._global_best._fitness < self._hyperparams.max_fitness
        # print(self._global_best._fitness,self._hyperparams.max_fitness)
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
        """
        Use Multiprocessing to process individual genomes in independent python processes
         as well as joining the results when they are done. This method also triggers the evolution.
        """
        max_proc = max(mp.cpu_count()-1, 1)
        pool = mp.Pool(processes=max_proc)

        # generates jobs for generation
        results = {}
        for i in range(len(self._species)):
            for j in range(len(self._species[i]._members)):
                results[(i, j)] = pool.apply_async(
                    evaluator,
                    args=[self._species[i]._members[j]]+list(args),
                    kwds=kwargs
                )

        # collect the results
        for key in tqdm(results):
            genome = self._species[key[0]]._members[key[1]]
            genome.set_fitness(results[key].get()+self._hyperparams.fitness_offset)

        pool.close()
        pool.join()
        self.save_fitness_history()
        self.save_max_fitness_history()
        self.save_population_history()
        if self.should_evolve():
            self.evolve()
        else:
            self._current_fittest = self.find_current_fittest()
            self._generation += 1
        current_best = self.get_current_fittest()
        self.save_network_history(len(current_best.get_connections()))

    def get_all_time_fittest(self):
        """Return the genome with the highest global fitness score."""
        return self._global_best

    def get_population(self):
        """calculates the true population size."""
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
    
    def get_average_fitness(self):
        """Get the average fitness of the population."""
        fitnesses = []
        for s in self._species:
            for genome in s._members:
                if genome.get_fitness() != 0:
                    fitnesses.append(genome.get_fitness())
        if not fitnesses:
            return -1
        return np.mean(fitnesses)

    def get_maximum_fitness(self):
        """Get the average fitness of the population."""
        fitnesses = []
        for s in self._species:
            for genome in s._members:
                if genome.get_fitness() != 0:
                    fitnesses.append(genome.get_fitness())
        if not fitnesses:
            return -1
        return np.max(fitnesses)

    def save(self, filename: str):
        """Save an instance of the population to disk."""
        with open(filename+'.neat', 'wb') as _out:
            pickle.dump(self, _out, pickle.HIGHEST_PROTOCOL)
    
    def save_fitness_history(self):
        self._past_fitnesses.append(self.get_average_fitness())
    
    def get_fitness_history(self):
        return self._past_fitnesses
    
    def save_max_fitness_history(self):
        self._past_max_fitnesses.append(self.get_maximum_fitness())
    
    def get_max_fitness_history(self):
        return self._past_max_fitnesses
    
    def get_species_count(self):
        return len(self._species)
    
    def save_network_history(self, network_size):
        self._network_history.append(network_size)
    
    def get_network_history(self):
        return self._network_history
    
    def save_population_history(self):
        self._population_history.append(self.get_population())
    
    def get_population_history(self):
        return self._population_history
    
    @staticmethod
    def load(filename):
        """Return an instance of a population from disk."""
        with open(filename+'.neat', 'rb') as _in:
            return pickle.load(_in)

def genomic_distance(a: Node, b: Node, distance_weights: dict):
    """Calculate the distance between two genomes."""
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
    return t1 + t3 + t2 # t4 

def generate_visualized_network(genome: Genome, generation, path, NETWORK_WIDTH=480, HEIGHT=480):
    """Generate the positions/colors of the neural network nodes"""
    nodes = {}
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca()
    ax.axis('off')
    plt.title(f'Best Genome with fitness {genome.get_fitness()} Network')
    _nodes = genome.get_nodes()
    layer_count = defaultdict(lambda: -1 * genome._inputs)
    index = {}
    max_nodes_per_layer = genome._inputs
    for n in _nodes:
        layer_count[_nodes[n].layer] += 1
        if layer_count[_nodes[n].layer] == 0:
            layer_count[_nodes[n].layer] += 1
        index[n] = layer_count[_nodes[n].layer]
        max_nodes_per_layer = max( max_nodes_per_layer, layer_count[_nodes[n].layer] + genome._inputs )
    for number in _nodes:
        if genome.is_input(number):
            color = 'blue'
            x = 0.05*NETWORK_WIDTH
            y = HEIGHT/4 + HEIGHT/5 * number
        elif genome.is_output(number):
            color = 'red'
            x = NETWORK_WIDTH-0.05*NETWORK_WIDTH
            y = HEIGHT/2
        else:
            color = 'black'
            x = NETWORK_WIDTH/10 + NETWORK_WIDTH/12 * _nodes[number].layer
            t = max( (layer_count[_nodes[number].layer]) * 2.5, max_nodes_per_layer)
            y = HEIGHT/2 + (HEIGHT / t) * index[number]
        nodes[number] = [(x, y), color]

    genes = genome.get_connections()
    sorted_innovations = sorted(genes.keys())
    for innovation in sorted_innovations:
        connection = genes[innovation]
        i, j = connection.in_node.number, connection.out_node.number
        if connection.enabled:
            color = 'green'
        else:
            color = 'red'
        x_values = [nodes[i][0][0], nodes[j][0][0]]
        y_values = [nodes[i][0][1], nodes[j][0][1]]
        ax.plot(x_values, y_values, color=color)

    for n in nodes:
        circle = plt.Circle(nodes[n][0], 5, color=nodes[n][1])
        ax.add_artist(circle)
    network_path = os.path.join(path,'networks')
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(network_path):
        os.makedirs(network_path)
    file_path = os.path.join(network_path, f'{generation}_network.png') 
    plt.savefig(file_path)
    plt.close()
