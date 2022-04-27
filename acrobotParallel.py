# To use render environment need to use only one cpu in neato and uncomment render part
#  or it will become frozen. (Probably way to fix this....)
from collections import defaultdict
import os
import pickle
import random
import numpy as np
#import cart_pole
from collections import defaultdict
import gym
import os
import sys
import matplotlib.pyplot as plt
# import pygame
sys.path.append('./neato')
from neato.genome import Genome
from neato.neato import NeatO
from neato.hyperparameters import Hyperparameters, tanh

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

def evaluate(genome:Genome):
    """Evaluates the current genome."""
    try:
        fitnesses = []
        for i in range(5):
            env = gym.make("Acrobot-v1")
            last_observation = env.reset()
            fitness = 0.
            done = False
            while not done:
                output = genome.forward(last_observation)[0]
                if output <= 0.3:
                    action = -1
                elif 0.3 < output <= 0.6:
                    action = 0
                else:
                    action = 1
                next_observation, reward, done, info = env.step(action)
                fitness += reward
                last_observation = next_observation
            fitnesses.append(fitness)
        sys.stdout.flush()
        return np.mean(fitnesses)
    except Exception as e:
        print('Evaluate','='*100)
        print(e)
        return 0


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
    if not os.path.exists('acrobat/acrobat_graphs/'):
        os.makedirs('acrobat/acrobat_graphs')
    if not os.path.exists('acrobat/acrobat_graphs/networks'):
        os.makedirs('acrobat/acrobat_graphs/networks')
    plt.savefig(f'acrobat/acrobat_graphs/networks/{generation}._network.png')
    plt.close()


def run():

    hyperparams = Hyperparameters()
    #hyperparams.max_generations = 300
    hyperparams.default_activation = tanh
    hyperparams.delta_threshold = 2
    hyperparams.mutation_probabilities['node'] = 0.2
    hyperparams.mutation_probabilities['connection'] = 0.2
    hyperparams.mutation_probabilities['mutate'] = 0.3
    hyperparams.mutation_probabilities['weight_perturb'] = 0.8
    hyperparams.mutation_probabilities['weight_set'] = 0.01
    hyperparams.mutation_probabilities['bias_perturb'] = 0.7
    hyperparams.mutation_probabilities['bias_set'] = 0.00
    hyperparams.mutation_probabilities['re-enable'] = 0.01
    hyperparams.fitness_offset = 1000
    hyperparams.max_fitness = 1000
    hyperparams.max_generations = 100
    hyperparams.survival_percentage = 0.25

    inputs = 6
    outputs = 1
    hidden_layers = 9
    population = 500
    if os.path.isfile('neato_acrobat.neat'):
        neato = NeatO.load('neato_acrobat')
        neato._hyperparams = hyperparams
    else:
        neato = NeatO(inputs, outputs, hidden_layers, population, hyperparams)
        neato.initialize()
        print(hyperparams.max_fitness)

    
    current_best = None
    print("Training...")
    # while neato.get_generation() < hyperparams.max_generations:
    while neato.should_evolve():
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
            neato.save('neato_acrobat')
        except Exception as e:
            print("Failed to save current neato:")
            print(e)
        try:
            generate_visualized_network(current_best, current_gen)
            with open(f'acrobat/models/neato_acrobat_{current_gen}_{abs(round(current_best.get_fitness()))}', 'wb') as f:
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
            plt.savefig(f'acrobat/acrobat_graphs/progress.png')
            plt.close()
            plt.figure()
            plt.plot(neato.get_network_history(), label='network size')
            plt.legend()
            plt.savefig(f'acrobat/acrobat_graphs/network_progress.png')
            plt.close()
            plt.figure()
            plt.plot(neato.get_population_history(), label='population size')
            plt.legend()
            plt.savefig(f'acrobat/acrobat_graphs/population_progress.png')
            plt.close()
        except Exception as e:
            print('progress plots', '='*40)
            print(e)
            print('='*100)
        # break

    with open('acrobat/neato_acrobat_best_individual', 'wb') as f:
        pickle.dump(neato._global_best, f)
    

if __name__ == '__main__':
    run()
