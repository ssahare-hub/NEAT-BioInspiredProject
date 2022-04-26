import multiprocessing
import os
import pickle

from collections import defaultdict
import numpy as np
import gym
import math
import os
import sys
import matplotlib.pyplot as plt

from neato.genome import Genome
sys.path.append('./neato')
from neato.neato import NeatO
from neato.hyperparameters import Hyperparameters, tanh, sigmoid
 
# Constants
WIDTH, HEIGHT = 640, 480
NETWORK_WIDTH = 480

# Flags
AI = True
DRAW_NETWORK = True

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)

runs_per_net=4
def evaluate(genome: Genome):
    """Evaluates the current genome."""
    fitnesses = []
    for i in range(runs_per_net):
        env = gym.make("CartPole-v1")
        env._max_episode_steps = 950
        observation = env.reset()

        fitness = 0.
        done = False
        while not done:
            action = genome.forward(observation)[0]
            #print(action)
            observation, reward, done, info = env.step(action <= 0.5)
            fitness += reward
        fitnesses.append(fitness)
    sys.stdout.flush()
    return np.mean(fitnesses)

def generate_visualized_network(genome: Genome, generation):
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
    if not os.path.exists('cartpole/cartpole_graphs'):
        os.makedirs('cartpole/cartpole_graphs')
    plt.savefig(f'cartpole/cartpole_graphs/{generation}._network.png')
    plt.close(fig)
    # plt.show()

def run():

    hyperparams = Hyperparameters()
    hyperparams.default_activation = sigmoid
    #hyperparams.max_generations = 300
    hyperparams.delta_threshold = 1.2
    hyperparams.mutation_probabilities['node'] = 0.2
    hyperparams.mutation_probabilities['connection'] = 0.2
    hyperparams.mutation_probabilities['mutate'] = 0.2
    hyperparams.mutation_probabilities['weight_perturb'] = 0.8
    hyperparams.mutation_probabilities['weight_set'] = 0.01
    hyperparams.mutation_probabilities['bias_perturb'] = 0.7
    hyperparams.mutation_probabilities['bias_set'] = 0.00
    hyperparams.mutation_probabilities['re-enable'] = 0.01
    hyperparams.survival_percentage = 0.2
    hyperparams.max_fitness = 949
    hyperparams.max_generations = 100

    inputs = 4
    outputs = 1
    hidden_layers = 6
    population = 1000
    
    if os.path.isfile('neato_cartpole.neat'):
            neato = NeatO.load('neato_cartpole')
            neato._hyperparams = hyperparams
    else:    
            neato = NeatO(inputs, outputs, hidden_layers, population, hyperparams)
            neato.initialize()
            print(hyperparams.max_fitness)

    current_best = None
    print("Training...")
    #while neato.should_evolve():
    while neato.get_generation() < hyperparams.max_generations:
        try:
            neato.evaluate_parallel(evaluate)

            # Print training progress
            current_gen = neato.get_generation()
            neato.save_fitness_history()
            neato.save_max_fitness_history()
            current_best = neato.get_current_fittest()
            mean_fitness = neato.get_average_fitness()
            neato.save_network_history(len(current_best.get_connections()))
            neato.save_population_history()
            print(
                "Mean Fitness: {} | Best Gen Fitness: {} | Species Count: {} |  Current gen: {}".format(
                    mean_fitness,
                    current_best.get_fitness(), 
                    neato.get_species_count(),
                    current_gen, 
                )
            )
            sys.stdout.flush()
            print('saving current population')
        except Exception as e:
            print('pre-saving', '-'*100)
            print(e)
        try:
            neato.save('neato_cartpole')
        except Exception as e:
            print("Failed to save current neato:")
            print(e)
        try:
            generate_visualized_network(current_best, current_gen)
            # NOTE: I wanted to see intermediate results
            # so saving genome whenever it beats the last best
            if current_best.get_fitness() > neato._global_best.get_fitness():
                with open(f'cartpole/neato_cartpole_best_individual_gen{current_gen}', 'wb') as f:
                    pickle.dump(current_best, f)
            neato.update_fittest()
        except Exception as e:
            print('Network', '='*40)
            print(e)
            print('='*100)
        # break
        try:
            plt.figure()
            plt.title('fitness over generations')
            plt.plot(neato.get_fitness_history(),label='average')
            plt.plot(neato.get_max_fitness_history(), label='max')
            plt.legend()
            plt.savefig(f'cartpole/cartpole_graphs/progress.png')
            plt.close()
            plt.figure()
            plt.plot(neato.get_network_history(), label='network size')
            plt.legend()
            plt.savefig(f'cartpole/cartpole_graphs/network_progress.png')
            plt.close()
            plt.figure()
            plt.plot(neato.get_population_history(), label='population size')
            plt.legend()
            plt.savefig(f'cartpole/cartpole_graphs/population_progress.png')
            plt.close()
        except Exception as e:
            print('progress plots', '='*40)
            print(e)
            print('='*100)


    with open('cartpole/neato_cartpole_best_individual', 'wb') as f:
        pickle.dump(neato._global_best, f)

if __name__ == '__main__':
    run()
