import os
import pickle
import math
from collections import defaultdict
import numpy as np
import gym
import os
import sys
import matplotlib.pyplot as plt
sys.path.append('./neato')
from neato.genome import Genome
from neato.brain import Brain
from neato.hyperparameters import Hyperparameters, tanh

EPISODE_DURATION = 500
 
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

def evaluate(genome):
    """Evaluates the current genome."""
    fitnesses = []
    for _ in range(5):
        env = gym.make("MountainCarContinuous-v0")
        # env._max_episode_steps = EPISODE_DURATION
        last_observation = env.reset()

        fitness = 0.
        done = False
        while not done:
            #if i == 0:
            #    env.render()
            action = genome.forward(last_observation)

            next_observation, reward, done, info = env.step(action)
            reward = reward = 100*((math.sin(3*next_observation[0]) * 0.0025 + 0.5 * next_observation[1] * next_observation[1]) - (math.sin(3*last_observation[0]) * 0.0025 + 0.5 * last_observation[1] * last_observation[1]))
            fitness += reward
            last_observation = next_observation
            # fitness += 1 if abs(observation[1]) >= 0.05 else 0
            # add a positive reward for being alive
        fitnesses.append(fitness)
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

    # print(len(_nodes))
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
        # t = txt.Text(nodes[n][0][0] + 10, nodes[n][0][1], str(genome._nodes[n].layer))
        # ax.add_artist(t)
        # t = txt.Text(nodes[n][0][0] - 10, nodes[n][0][1] + 10, str(n), color='red')
        # ax.add_artist(t)
    if not os.path.exists('mountaincar_graphs'):
        os.makedirs('mountaincar_graphs')
    plt.savefig(f'mountaincar_graphs/{generation}._network.png')
    plt.close()

def run():

    hyperparams = Hyperparameters()
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
    hyperparams.perturbation_range['weight_perturb_max'] = 2
    hyperparams.perturbation_range['weight_perturb_min'] = -2

    hyperparams.fitness_offset = 0
    hyperparams.max_fitness = hyperparams.fitness_offset+0.7029
    hyperparams.max_generations = 300
    hyperparams.distance_weights['matching_connections'] = 0.75

    inputs = 2
    outputs = 1
    hidden_layers = 6
    population = 1000
    if os.path.isfile('mountaincar.neat'):
        brain = Brain.load('mountaincar')
        brain._hyperparams = hyperparams
    else:    
        brain = Brain(inputs, outputs, hidden_layers, population, hyperparams)
        brain.generate()
        print(hyperparams.max_fitness)

    current_best = None
    print("Training...")
    fitness_history = []
    while brain.should_evolve():
        brain.evaluate_parallel(evaluate)

        # Print training progress
        current_gen = brain.get_generation()
        brain.update_fittest()
        current_best = brain.get_current_fittest()
        mean_fitness = brain.get_average_fitness()
        brain.save_fitness_history()
        brain.save_max_fitness_history()
        print(
            "Mean Fitness: {} | Current Accuracy: {} | Species Count: {} | Population count: {} | Current gen: {}".format(
                mean_fitness,
                current_best.get_fitness(), 
                brain.get_species_count(), 
                brain.get_population(),
                current_gen
            )
        )
        sys.stdout.flush()
        print('saving current population')
        brain.save('mountaincar')
        generate_visualized_network(current_best, current_gen)
        # NOTE: I wanted to see intermediate results
        # so saving genome whenever it beats the last best
        if current_best.get_fitness() >= brain._global_best.get_fitness():
            with open(f'mountaincar_best_individual_gen{current_gen}', 'wb') as f:
                pickle.dump(current_best, f)
        brain.update_fittest()
        # break
        try:
            plt.figure()
            plt.title('fitness over generations')
            plt.plot(brain.get_fitness_history(),label='average')
            plt.plot(brain.get_max_fitness_history(), label='max')
            plt.legend()
            plt.savefig(f'mountaincar_graphs/progress.png')
            plt.close()
            plt.figure()
            plt.plot(brain.get_network_history(), label='network size')
            plt.legend()
            plt.savefig(f'mountaincar_graphs/network_progress.png')
            plt.close()
            plt.figure()
            plt.plot(brain.get_population_history(), label='population size')
            plt.legend()
            plt.savefig(f'mountaincar_graphs/population_progress.png')
            plt.close()
        except Exception as e:
            print('progress plots', '='*40)
            print(e)
            print('='*100)

    print('done')
    with open('mountaincar_best_individual', 'wb') as f:
        pickle.dump(brain.get_all_time_fittest(), f)
    

if __name__ == '__main__':
    run()
