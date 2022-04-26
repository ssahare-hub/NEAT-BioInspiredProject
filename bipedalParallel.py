import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import gym
import os
import sys
sys.path.append('./neato')
from neato.genome import Genome
from neato.neato import NeatO
from neato.hyperparameters import Hyperparameters, tanh, sigmoid
from collections import defaultdict

EPISODE_DURATION = 500
seed = 0
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

def evaluate(genome: Genome):
    """Evaluates the current genome."""
    fitnesses = []
    for i in range(2):
        env = gym.make("BipedalWalker-v3")
        env.reset(seed=seed)
        #env._max_episode_steps = EPISODE_DURATION
        observation = env.reset()

        fitness = 0.
        done = False
        while not done:
            action = genome.forward(observation)
            observation, reward, done, info = env.step(action)
            fitness += reward
        fitnesses.append(fitness+100)
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
    if not os.path.exists('bipedal/bipedal_graphs'):
        os.makedirs('bipedal/bipedal_graphs')
    plt.savefig(f'bipedal/bipedal_graphs/{generation}._network.png')
    plt.close(fig)
    # plt.show()

def run():

    hyperparams = Hyperparameters()
    hyperparams.max_generations = 50
    hyperparams.default_activation = tanh
    hyperparams.delta_threshold = 1.5
    hyperparams.mutation_probabilities['node'] = 0.05
    hyperparams.mutation_probabilities['connection'] = 0.05
    hyperparams.mutation_probabilities['mutate'] = 0.05
    hyperparams.mutation_probabilities['weight_perturb'] = 0.8
    hyperparams.mutation_probabilities['weight_set'] = 0.01
    hyperparams.mutation_probabilities['bias_perturb'] = 0.8
    hyperparams.mutation_probabilities['bias_set'] = 0.01
    hyperparams.mutation_probabilities['re-enable'] = 0.01
    hyperparams.fitness_offset = 500
    hyperparams.survival_percentage = 0.2
    hyperparams.max_fitness = hyperparams.fitness_offset + 300
    hyperparams.max_generations = 100

    inputs = 24
    outputs = 4
    hidden_layers = 6
    population = 200
    
    if os.path.isfile('neato_bipedal.neat'):
        neato = NeatO.load('neato_bipedal')
        neato._hyperparams = hyperparams
    else:    
        neato = NeatO(inputs, outputs, hidden_layers, population, hyperparams)
        neato.initialize()
        print(hyperparams.max_fitness)

    current_best = None
    print("Training...")
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
            neato.save('neato_bipedal')
        except Exception as e:
            print("Failed to save current neato:")
            print(e)
        try:
            generate_visualized_network(current_best, current_gen)
            # NOTE: I wanted to see intermediate results
            # so saving genome whenever it beats the last best
            if current_best.get_fitness() > neato._global_best.get_fitness():
                with open(f'bipedal/neato_bipedal_best_individual_gen{current_gen}', 'wb') as f:
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
            plt.savefig(f'bipedal/bipedal_graphs/progress.png')
            plt.close()
            plt.figure()
            plt.plot(neato.get_network_history(), label='network size')
            plt.legend()
            plt.savefig(f'bipedal/bipedal_graphs/network_progress.png')
            plt.close()
            plt.figure()
            plt.plot(neato.get_population_history(), label='population size')
            plt.legend()
            plt.savefig(f'bipedal/bipedal_graphs/population_progress.png')
            plt.close()
        except Exception as e:
            print('progress plots', '='*40)
            print(e)
            print('='*100)


    with open('bipedal/bipedal_cartpole_best_individual', 'wb') as f:
        pickle.dump(neato._global_best, f)

if __name__ == '__main__':
    run()
