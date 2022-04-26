# pip3 install gym
# for gym stuff: 
# apt install xvfb ffmpeg xorg-dev libsdl2-dev swig cmake
# pip3 install gym[box2d]



# To use render environment need to use only one cpu in brain and uncomment render part
#  or it will become frozen. (Probably way to fix this....)
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
            #if i == 0:
            #    env.render()
            action = genome.forward(observation)[0]
            #print(action)
            observation, reward, done, info = env.step(action <= 0.5)
            fitness += reward
        #env.close()
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
    if not os.path.exists('cartpole_graphs'):
        os.makedirs('cartpole_graphs')
    if not os.path.exists('cartpole_best_individuals'):
        os.makedirs('cartpole_best_individuals')
    plt.savefig(f'cartpole_graphs/{generation}._network.png')
    plt.close(fig)
    # plt.show()

def run():

    hyperparams = Hyperparameters()
    hyperparams.default_activation = sigmoid
    #hyperparams.max_generations = 300
    hyperparams.delta_threshold = 1.2
    hyperparams.mutation_probabilities['node'] = 0.05
    hyperparams.mutation_probabilities['connection'] = 0.05
    hyperparams.mutation_probabilities['mutate'] = 0.05
    hyperparams.max_fitness = 949
    hyperparams.max_generations = 10

    inputs = 4
    outputs = 1
    hidden_layers = 6
    population = 1000
    
    if os.path.isfile('cartpole.neat'):
            brain = NeatO.load('cartpole')
            brain._hyperparams = hyperparams
    else:    
            brain = NeatO(inputs, outputs, hidden_layers, population, hyperparams)
            brain.initialize()
            print(hyperparams.max_fitness)

    current_best = None
    print("Training...")
    while brain.should_evolve:
        brain.evaluate_parallel(evaluate)

        # Print training progress
        current_gen = brain.get_generation()
        brain.save_fitness_history()
        brain.save_max_fitness_history()
        current_best = brain.get_current_fittest()
        mean_fitness = brain.get_average_fitness()
        brain.save_network_history(len(current_best.get_connections()))
        brain.save_population_history()
        print(
            "Mean Fitness: {} | Best Gen Fitness: {} | Species Count: {} |  Current gen: {}".format(
                mean_fitness,
                current_best.get_fitness(), 
                brain.get_species_count(),
                current_gen, 
            )
        )
        sys.stdout.flush()
        print('saving current population')
        brain.save('cartpole_best_individuals/cartpole')
        try:
            generate_visualized_network(current_best, current_gen)
        except Exception as e:
            print('Network', '='*40)
            print(e)
            print('='*100)
        # NOTE: I wanted to see intermediate results
        # so saving genome whenever it beats the last best
        if current_best.get_fitness() > brain._global_best.get_fitness():
            with open(f'cartpole_best_individuals/cartpole_best_individual_gen{current_gen}', 'wb') as f:
                pickle.dump(current_best, f)
        brain.update_fittest()
        # break
        try:
            plt.figure()
            plt.title('fitness over generations')
            plt.plot(brain.get_fitness_history(),label='average')
            plt.plot(brain.get_max_fitness_history(), label='max')
            plt.legend()
            plt.savefig(f'cartpole_graphs/progress.png')
            plt.close()
            plt.figure()
            plt.plot(brain.get_network_history(), label='network size')
            plt.legend()
            plt.savefig(f'cartpole_graphs/network_progress.png')
            plt.close()
            plt.figure()
            plt.plot(brain.get_population_history(), label='population size')
            plt.legend()
            plt.savefig(f'cartpole_graphs/population_progress.png')
            plt.close()
        except Exception as e:
            print('progress plots', '='*40)
            print(e)
            print('='*100)


    with open('cartpole_best_individuals/cartpole_best_individual', 'wb') as f:
        pickle.dump(brain._global_best, f)

if __name__ == '__main__':
    run()
