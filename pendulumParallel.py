# To use render environment need to use only one cpu in brain and uncomment render part
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

EPISODE_DURATION = 200
 
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
            env = gym.make("Pendulum-v1")
            env._max_episode_steps = EPISODE_DURATION
            last_observation = env.reset()
            fitness = 0.
            done = False
            while not done:
                action = genome.forward(last_observation)[0]
                next_observation, reward, done, info = env.step([action * 2])
                reward = 25* np.exp(-1*(next_observation[0]-1)*(next_observation[0]-1)/0.001)-100*np.abs(10*0.5 - (10*0.5*next_observation[0] + 0.5*0.3333*next_observation[2] * next_observation[2])) + 100*np.abs(10*0.5 - (10*0.5*last_observation[0] + 0.5*0.3333*last_observation[2] * last_observation[2]))
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
    if not os.path.exists('pendulum_graphs'):
        os.makedirs('pendulum_graphs')
    plt.savefig(f'pendulum_graphs/{generation}._network.png')
    plt.close()


def run():

    hyperparams = Hyperparameters()
    #hyperparams.max_generations = 300
    hyperparams.default_activation = tanh
    hyperparams.delta_threshold = 2
    hyperparams.mutation_probabilities['node'] = 0.2
    hyperparams.mutation_probabilities['connection'] = 0.2
    hyperparams.mutation_probabilities['mutate'] = 0.75
    hyperparams.mutation_probabilities['weight_perturb'] = 0.8
    hyperparams.mutation_probabilities['weight_set'] = 0.01
    hyperparams.mutation_probabilities['bias_perturb'] = 0.8
    hyperparams.mutation_probabilities['bias_set'] = 0.01
    hyperparams.mutation_probabilities['re-enable'] = 0.01
    hyperparams.fitness_offset = 0 * EPISODE_DURATION
    hyperparams.max_fitness = 8000
    hyperparams.max_generations = 300
    hyperparams.survival_percentage = 0.50

    inputs = 3
    outputs = 1
    hidden_layers = 12
    population = 300
    if os.path.isfile('neato_pendulum.neat'):
        brain = NeatO.load('neato_pendulum')
        brain._hyperparams = hyperparams
    else:
        brain = NeatO(inputs, outputs, hidden_layers, population, hyperparams)
        brain.initialize()
        print(hyperparams.max_fitness)

    
    current_best = None
    print("Training...")
    # while brain.get_generation() < hyperparams.max_generations:
    while brain.should_evolve():
        try:
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
        except Exception as e:
            print('pre-saving', '-'*100)
            print(e)
        try:
            brain.save('neato_pendulum')
        except Exception as e:
            print("Failed to save current brain:")
            print(e)
        try:
            generate_visualized_network(current_best, current_gen)
            # NOTE: I wanted to see intermediate results
            # so saving genome whenever it beats the last best
            if current_best.get_fitness() > brain._global_best.get_fitness():
                with open(f'pendulum/neato_pendulum_best_individual_gen{current_gen}', 'wb') as f:
                    pickle.dump(current_best, f)
            brain.update_fittest()
        except Exception as e:
            print('Network', '='*40)
            print(e)
            print('='*100)
        # break
        try:
            plt.figure()
            plt.title('fitness over generations')
            plt.plot(brain.get_fitness_history(),label='average')
            plt.plot(brain.get_max_fitness_history(), label='max')
            plt.legend()
            plt.savefig(f'pendulum/pendulum_graphs/progress.png')
            plt.close()
            plt.figure()
            plt.plot(brain.get_network_history(), label='network size')
            plt.legend()
            plt.savefig(f'pendulum/pendulum_graphs/network_progress.png')
            plt.close()
            plt.figure()
            plt.plot(brain.get_population_history(), label='population size')
            plt.legend()
            plt.savefig(f'pendulum/pendulum_graphs/population_progress.png')
            plt.close()
        except Exception as e:
            print('progress plots', '='*40)
            print(e)
            print('='*100)
        # break

    with open('pendulum/neato_pendulum_best_individual', 'wb') as f:
        pickle.dump(brain._global_best, f)
    

if __name__ == '__main__':
    run()
